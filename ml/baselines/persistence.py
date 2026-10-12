"""
Persistence Heat Risk Baseline (Evaluation-Only Benchmark).

CRITICAL SCIENTIFIC PURPOSE:
This is NOT an ML model and involves zero parameter training or gradient descent.
Its purpose is to serve as the fundamental benchmark for evaluating future forecasting models:
"Does a trained ML model actually forecast better than simply assuming current heat risk persists?"

Baseline Heuristic:
-------------------
At prediction time t:
    predicted_future_risk(t, t + H] = instantaneous_risk(t)

Under diurnal solar heating, persistence systematically underestimates afternoon peak heat
when predicting from morning observations (e.g. 09:00 -> 15:00), making it the essential
reference point to prove that a forward ML forecaster captures thermal dynamics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.features.weather_features import calculate_heat_index, calculate_wet_bulb_temperature
from ml.pipelines.target import ThresholdBasedHeatTargetGenerator


class PersistenceHeatBaseline:
    """
    Evaluation-only benchmark that predicts future 6-hour risk equals current risk at time t.
    """

    CATEGORIES = ["LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"]

    def __init__(self, target_generator: Optional[ThresholdBasedHeatTargetGenerator] = None):
        self.target_generator = target_generator or ThresholdBasedHeatTargetGenerator()

    def predict_row(self, row: pd.Series) -> Optional[str]:
        """
        Predict future risk category for a single feature row using only observations at time t.
        """
        # Read or compute instantaneous heat index & wet-bulb at t
        hi = row.get("heat_index_c")
        wb = row.get("wet_bulb_temperature_c") or row.get("wet_bulb_temp_c")

        if hi is None or pd.isna(hi) or wb is None or pd.isna(wb):
            t = row.get("temperature_c")
            rh = row.get("relative_humidity_pct")
            if t is not None and rh is not None and not pd.isna(t) and not pd.isna(rh):
                hi = calculate_heat_index(float(t), float(rh))
                wb = calculate_wet_bulb_temperature(float(t), float(rh))

        return self.target_generator.map_metrics_to_category(
            float(hi) if hi is not None and not pd.isna(hi) else None,
            float(wb) if wb is not None and not pd.isna(wb) else None,
        )

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """
        Generate persistence predictions for every row in feature_df.
        Uses ONLY information available at timestamp t.
        """
        preds = feature_df.apply(self.predict_row, axis=1)
        preds.name = "persistence_pred"
        return preds

    def evaluate(self, feature_df: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Evaluate persistence baseline performance against ground-truth forward target.
        Calculates accuracy, per-class metrics, confusion matrix, and transition analysis.
        """
        y_pred = self.predict(feature_df)

        valid_mask = y_true.notna() & y_pred.notna()
        y_t = y_true[valid_mask].astype(str)
        y_p = y_pred[valid_mask].astype(str)

        total_samples = len(y_t)
        if total_samples == 0:
            return {"error": "No valid ground truth and prediction pairs to evaluate."}

        correct = (y_t == y_p).sum()
        accuracy = float(correct / total_samples)

        # Per-class metrics
        class_metrics = {}
        for cat in self.CATEGORIES:
            tp = int(((y_t == cat) & (y_p == cat)).sum())
            fp = int(((y_t != cat) & (y_p == cat)).sum())
            fn = int(((y_t == cat) & (y_p != cat)).sum())
            support = int((y_t == cat).sum())

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            class_metrics[cat] = {
                "support": support,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        macro_f1 = float(np.mean([m["f1"] for m in class_metrics.values()]))

        # Confusion Matrix
        confusion = {}
        for actual in self.CATEGORIES:
            confusion[actual] = {}
            for pred in self.CATEGORIES:
                confusion[actual][pred] = int(((y_t == actual) & (y_p == pred)).sum())

        # Transition Analysis (The core reason forecasting is needed)
        cat_rank = {cat: i for i, cat in enumerate(self.CATEGORIES)}
        transitions_total = 0
        upward_escalations = 0
        downward_relaxations = 0

        for true_val, pred_val in zip(y_t, y_p):
            t_rank = cat_rank.get(true_val, -1)
            p_rank = cat_rank.get(pred_val, -1)
            if t_rank != p_rank:
                transitions_total += 1
                if t_rank > p_rank:
                    upward_escalations += 1
                else:
                    downward_relaxations += 1

        escalation_miss_rate = float(upward_escalations / total_samples)

        return {
            "total_evaluated": total_samples,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "class_metrics": class_metrics,
            "confusion_matrix": confusion,
            "transition_analysis": {
                "total_transitions": transitions_total,
                "transition_percentage": round(100.0 * transitions_total / total_samples, 2),
                "upward_escalations_missed_by_persistence": upward_escalations,
                "downward_relaxations_missed_by_persistence": downward_relaxations,
                "escalation_miss_rate": round(escalation_miss_rate, 4),
            },
            "scientific_takeaway": (
                f"Persistence achieved {round(accuracy * 100, 2)}% accuracy, but failed on "
                f"{transitions_total} dynamic transitions ({round(100.0 * transitions_total / total_samples, 2)}% of time steps). "
                f"Specifically, persistence missed {upward_escalations} upward heat escalations where risk "
                f"intensified during the 6-hour forward window. Future ML models must demonstrate higher recall on these escalations."
            ),
        }
