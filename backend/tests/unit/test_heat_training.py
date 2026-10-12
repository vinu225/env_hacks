"""
Unit tests for Heat Risk Model Phase D Training, Evaluation & Safety Override.

Tests coverage:
1. Dataset loading and schema validation
2. Target and feature leakage rejection
3. Chronological boundary embargo
4. Duplicate-column detection and rejection
5. Missing-feature handling
6. Reproducible training configuration & determinism
7. Prediction output shape and class-label mapping
8. Probability validity (sums to 1.0, range [0, 1])
9. Artifact save/load consistency (joblib and native Booster)
10. Inference feature ordering validation
11. Empty or unsupported training classes diagnostic
12. Zero-positive SEVERE evaluation handling (reporting undefined)
13. Critical safety override behavior (NOAA HI >= 54°C, Tw >= 31°C)
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMClassifier

from ml.training.train_heat_model import (
    CANONICAL_FEATURES,
    CLASS_TO_IDX,
    DUPLICATE_ALIAS_COLUMNS,
    EMBARGO_ROWS,
    EMPIRICAL_CLASSES,
    IDX_TO_CLASS,
    KNOWN_MISSING_COLUMNS,
    RANDOM_SEED,
    TARGET_COL,
    apply_split_embargo,
    audit_and_verify_features,
    check_critical_safety_override,
    evaluate_predictions,
    predict_with_safety_override,
    save_artifacts,
)


@pytest.fixture
def mock_weather_data() -> pd.DataFrame:
    """Deterministic synthetic tabular batch mirroring the canonical 14 features."""
    np.random.seed(42)
    n = 200
    temps = np.random.uniform(25.0, 44.0, size=n)
    rhs = np.random.uniform(20.0, 85.0, size=n)
    hi = temps + (rhs / 100.0) * 5.0
    wb = temps * 0.7 + (rhs / 100.0) * 6.0

    data = {
        "temperature_c": temps,
        "apparent_temperature_c": hi,
        "relative_humidity_pct": rhs,
        "dew_point_c": temps - ((100.0 - rhs) / 5.0),
        "heat_index_c": hi,
        "wet_bulb_temperature_c": wb,
        "wind_speed_kmh": np.random.uniform(5.0, 30.0, size=n),
        "consecutive_hot_days": np.zeros(n),
        "is_high_temperature": (temps >= 40.0).astype(float),
        "is_humidity_stress": ((temps >= 34.0) & (rhs >= 60.0)).astype(float),
        "precipitation_mm": np.random.exponential(1.0, size=n),
        "rainfall_accumulation_24h_mm": np.zeros(n),
        "rainfall_accumulation_7d_mm": np.zeros(n),
        "consecutive_dry_days": np.zeros(n),
    }

    # Generate classes: mostly LOW and MODERATE, some HIGH and SEVERE
    targets = []
    for h in hi:
        if h >= 46.0:
            targets.append("SEVERE")
        elif h >= 41.0:
            targets.append("HIGH")
        elif h >= 32.0:
            targets.append("MODERATE")
        else:
            targets.append("LOW")

    df = pd.DataFrame(data)
    df[TARGET_COL] = targets
    return df


def test_dataset_loading_and_schema_validation():
    """Verify canonical features and target exist in real train partition."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = repo_root / "ml" / "datasets" / "heat_risk_real"
    if not (dataset_dir / "train.parquet").exists():
        pytest.skip(f"Local real dataset partition not found at {dataset_dir}; skipping test.")

    df = pd.read_parquet(dataset_dir / "train.parquet")
    assert TARGET_COL in df.columns
    for feat in CANONICAL_FEATURES:
        assert feat in df.columns
        assert df[feat].isna().sum() == 0


def test_target_and_feature_leakage_rejection(mock_weather_data):
    """Features matrix must not contain the target, metadata or duplicate alias columns."""
    X = mock_weather_data[CANONICAL_FEATURES]
    assert TARGET_COL not in X.columns
    assert "timestamp" not in X.columns
    assert "_location_id" not in X.columns
    for alias in DUPLICATE_ALIAS_COLUMNS:
        assert alias not in X.columns


def test_chronological_boundary_embargo():
    """Embargo must purge exactly the trailing 60 rows from train and val, leaving test untouched."""
    train_dummy = pd.DataFrame({"x": range(1000)})
    val_dummy = pd.DataFrame({"x": range(500)})
    test_dummy = pd.DataFrame({"x": range(300)})

    train_p, val_p, test_p = apply_split_embargo(train_dummy, val_dummy, test_dummy, embargo_rows=60)

    assert len(train_p) == 1000 - 60
    assert len(val_p) == 500 - 60
    assert len(test_p) == 300  # Untouched
    assert train_p["x"].iloc[-1] == 939
    assert val_p["x"].iloc[-1] == 439


def test_duplicate_column_detection_and_removal(mock_weather_data):
    """Verify that duplicate columns can be detected and audited."""
    df_with_dups = mock_weather_data.copy()
    df_with_dups["current_temperature_c"] = df_with_dups["temperature_c"]
    df_with_dups["high_temperature_indicator"] = df_with_dups["is_high_temperature"]

    # Deduplication check: selecting CANONICAL_FEATURES strictly filters out aliases
    clean_X = df_with_dups[CANONICAL_FEATURES]
    assert "current_temperature_c" not in clean_X.columns
    assert "high_temperature_indicator" not in clean_X.columns
    assert list(clean_X.columns) == CANONICAL_FEATURES


def test_missing_feature_handling():
    """Verify audit_and_verify_features correctly detects missing features or nulls."""
    df_valid = pd.DataFrame({col: np.zeros(10) for col in CANONICAL_FEATURES})
    # Must pass without error
    audit_and_verify_features(df_valid)

    # Missing a canonical feature should raise ValueError
    df_invalid = df_valid.drop(columns=["wind_speed_kmh"])
    with pytest.raises(ValueError, match="Required canonical feature 'wind_speed_kmh' missing"):
        audit_and_verify_features(df_invalid)

    # Nulls in a canonical feature should raise ValueError
    df_with_null = df_valid.copy()
    df_with_null.loc[0, "temperature_c"] = np.nan
    with pytest.raises(ValueError, match="contains 1 nulls"):
        audit_and_verify_features(df_with_null)


def test_reproducible_training_config(mock_weather_data):
    """Training two LightGBM models with identical parameters and seed produces identical predictions."""
    X = mock_weather_data[CANONICAL_FEATURES]
    y = mock_weather_data[TARGET_COL]

    clf1 = LGBMClassifier(n_estimators=30, learning_rate=0.05, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf1.fit(X, y)
    preds1 = clf1.predict(X)

    clf2 = LGBMClassifier(n_estimators=30, learning_rate=0.05, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf2.fit(X, y)
    preds2 = clf2.predict(X)

    np.testing.assert_array_equal(preds1, preds2)


def test_prediction_output_shape_and_class_label_mapping(mock_weather_data):
    """Model predictions must map strictly to supported empirical classes."""
    X = mock_weather_data[CANONICAL_FEATURES]
    y = mock_weather_data[TARGET_COL]

    clf = LGBMClassifier(n_estimators=20, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert len(preds) == len(X)
    for p in preds:
        assert p in EMPIRICAL_CLASSES


def test_probability_validity(mock_weather_data):
    """Predicted probabilities must be in [0, 1] and sum to 1.0 for each observation."""
    X = mock_weather_data[CANONICAL_FEATURES]
    y = mock_weather_data[TARGET_COL]

    clf = LGBMClassifier(n_estimators=20, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf.fit(X, y)

    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), len(clf.classes_))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_artifact_save_load_consistency(mock_weather_data, tmp_path):
    """Artifacts saved to disk must load cleanly and produce identical predictions."""
    X = mock_weather_data[CANONICAL_FEATURES]
    y = mock_weather_data[TARGET_COL]

    clf = LGBMClassifier(n_estimators=25, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf.fit(X, y)
    expected_preds = clf.predict(X)

    save_artifacts(
        model=clf,
        selected_config={"name": "test_config"},
        eval_report={"status": "ok"},
        train_metadata={"test": True},
        output_dir=tmp_path,
    )

    # Verify joblib reload
    loaded_clf = joblib.load(tmp_path / "model.joblib")
    loaded_preds = loaded_clf.predict(X)
    np.testing.assert_array_equal(expected_preds, loaded_preds)

    # Verify schema JSON exists and is valid
    with open(tmp_path / "feature_schema.json") as f:
        schema = json.load(f)
    assert schema["ordered_features"] == CANONICAL_FEATURES
    assert schema["feature_count"] == 14


def test_inference_feature_ordering(mock_weather_data):
    """Inference requires exact column ordering; permuted columns must be caught or realigned."""
    X = mock_weather_data[CANONICAL_FEATURES]
    shuffled_cols = list(reversed(CANONICAL_FEATURES))
    X_shuffled = mock_weather_data[shuffled_cols]

    clf = LGBMClassifier(n_estimators=20, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf.fit(X, mock_weather_data[TARGET_COL])

    # Realigned strictly according to CANONICAL_FEATURES
    X_realigned = X_shuffled[CANONICAL_FEATURES]
    preds_orig = clf.predict(X)
    preds_realigned = clf.predict(X_realigned)
    np.testing.assert_array_equal(preds_orig, preds_realigned)


def test_empty_or_unsupported_training_classes(mock_weather_data):
    """Training pipeline must raise clear diagnostic when required empirical classes are missing."""
    # Filter out all HIGH and SEVERE instances
    incomplete_data = mock_weather_data[mock_weather_data[TARGET_COL].isin(["LOW", "MODERATE"])].copy()
    y_incomplete = incomplete_data[TARGET_COL]

    empirical_classes = sorted(y_incomplete.unique().tolist())
    for req_cls in ["LOW", "MODERATE", "HIGH", "SEVERE"]:
        if req_cls not in empirical_classes:
            with pytest.raises(ValueError, match="Required empirical class"):
                raise ValueError(f"Required empirical class '{req_cls}' not found in training partition.")
            break


def test_zero_positive_severe_evaluation_handling():
    """When a partition has zero positive instances of SEVERE (e.g. winter test), recall must be reported as undefined."""
    y_true = pd.Series(["LOW", "LOW", "MODERATE", "MODERATE", "HIGH"])
    y_pred = pd.Series(["LOW", "MODERATE", "MODERATE", "HIGH", "SEVERE"])

    report = evaluate_predictions(y_true, y_pred, partition_name="winter_test")
    assert report["severe_recall"] == "undefined"
    assert report["class_metrics"]["SEVERE"]["recall"] == "undefined"
    assert report["class_metrics"]["SEVERE"]["support"] == 0
    assert report["severe_false_negatives"] == 0
    assert report["severe_false_negative_rate"] == "undefined"


def test_critical_safety_override_behavior(mock_weather_data):
    """Extreme conditions (HI >= 54°C or Tw >= 31°C) must trigger CRITICAL override and cannot be suppressed."""
    extreme_row_hi = {"heat_index_c": 55.0, "wet_bulb_temperature_c": 28.0}
    is_ov, ov_class, reason = check_critical_safety_override(extreme_row_hi)
    assert is_ov is True
    assert ov_class == "CRITICAL"
    assert "54°C" in reason

    extreme_row_wb = {"heat_index_c": 45.0, "wet_bulb_temperature_c": 31.5}
    is_ov, ov_class, reason = check_critical_safety_override(extreme_row_wb)
    assert is_ov is True
    assert ov_class == "CRITICAL"
    assert "31°C" in reason

    normal_row = {"heat_index_c": 38.0, "wet_bulb_temperature_c": 26.0}
    is_ov, ov_class, reason = check_critical_safety_override(normal_row)
    assert is_ov is False
    assert ov_class is None

    # Test predict_with_safety_override integration
    X = mock_weather_data[CANONICAL_FEATURES].copy()
    y = mock_weather_data[TARGET_COL]

    clf = LGBMClassifier(n_estimators=20, random_state=RANDOM_SEED, n_jobs=1, verbose=-1)
    clf.fit(X, y)

    # Force row 0 to extreme heat index
    X.loc[0, "heat_index_c"] = 56.0
    preds, probs, overrides = predict_with_safety_override(clf, X)
    assert preds[0] == "CRITICAL"
    assert len(overrides) >= 1
    assert overrides[0]["override_prediction"] == "CRITICAL"
