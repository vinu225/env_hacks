# Machine Learning Subsystem Directory Structure

This directory houses the offline and training lifecycle for ML environmental risk models:

```
ml/
├── datasets/       # Raw, processed, and benchmark evaluation datasets
├── notebooks/      # Exploratory data analysis (EDA) and experimental feature modeling
├── training/       # Reproducible model training scripts (XGBoost, LightGBM, Random Forest)
├── evaluation/     # Backtesting, ROC/PR curves, Brier scores, and calibration checks
├── inference/      # Offline batch inference and model export utilities (ONNX, Joblib)
├── pipelines/      # Automated feature and training pipelines (e.g. AWS SageMaker Pipelines)
├── artifacts/      # Exported model weights, scalers, and metadata manifests
└── README.md
```

## Future ML Model Integration Workflow

1. **Experimental Phase (`notebooks/`)**:
   - Analyze historical heatwaves (e.g. IMD gridded datasets, ERA5 reanalysis).
   - Engineer rolling indicators (7-day heat index, dry spells, antecedent precipitation).

2. **Training & Validation (`training/`, `evaluation/`)**:
   - Train tabular models (e.g., Gradient Boosted Trees for non-linear thermal/drought risk).
   - Evaluate against ground-truth incident records and meteorological benchmark events.

3. **Artifact Packaging (`artifacts/`)**:
   - Export serialized weights (`.joblib` or `.onnx`) accompanied by a `manifest.json` describing:
     - `model_name`
     - `model_version`
     - `features` list and order
     - `train_metrics` (F1, AUC, Brier Score)

4. **Production Drop-in (`backend/app/models/<category>/model.py`)**:
   - Implement `RiskModel` contract (`MLHeatModel`, `MLWaterModel`, `MLRainfallModel`).
   - Wire into `<Category>Predictor`. Zero changes needed in API, frontend, or risk fusion layers.
