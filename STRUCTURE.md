# Repository Structure

## Insider Threat Detection — CERT r4.2

```
Insider-Threat-Detection-CERT-r4.2/
│
├── 01_setup_environment.py          # Verify file paths and dataset access
├── 02_load_data.py                  # Load logon, device, email, psychometric, insiders
├── 03_feature_engineering.py        # Aggregate per-user behavioral features
├── 04_merge_and_label.py            # Merge feature tables, create binary target label
├── 05_preprocessing.py              # Missing value imputation, split, scaling, SMOTE
├── 06_model_training.py             # Train Random Forest and XGBoost classifiers
├── 07_evaluation.py                 # ROC curve comparison
├── 08_shap_explainability.py        # SHAP summary, bar, and force plots
├── 09_threshold_tuning.py           # Precision/Recall/F1 threshold analysis
├── 10_cross_validation.py           # 5-fold stratified cross-validation
├── 11_save_outputs.py               # Save model, scaler, features, results summary
├── master_training_script.py        # End-to-end pipeline (single script reproduction)
│
├── notebooks/
│   └── Insider_Threat_Detection_CERT_r4.2.ipynb   # Kaggle notebook export
│
├── models/
│   ├── insider_threat_rf_model.pkl  # Trained Random Forest model
│   └── insider_threat_scaler.pkl    # Fitted StandardScaler
│
├── outputs/
│   ├── shap_summary.png             # SHAP beeswarm — feature impact direction
│   ├── shap_bar.png                 # SHAP mean absolute impact per feature
│   ├── shap_force.png               # SHAP force plot — single insider explanation
│   ├── roc_curve_comparison.png     # ROC curve — RF vs XGBoost vs baseline
│   ├── threshold_curve.png          # Precision/Recall/F1 vs threshold
│   ├── confusion_matrix_random_forest.png
│   └── confusion_matrix_xgboost.png
│
├── data/
│   └── README.md                    # Dataset access instructions (not uploaded)
│
├── feature_cols.json                # Ordered feature column list
├── results_summary.json             # Full project results record
├── README.md                        # Project documentation
├── STRUCTURE.md                     # This file
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Ignored files
```

## Script Execution Order

Run numbered scripts sequentially (01 through 11), or use `master_training_script.py`
to reproduce the full pipeline end-to-end from a single execution.

Each numbered script assumes the previous script's outputs are in memory
(designed for Kaggle notebook cell-by-cell execution).
The master script is self-contained and handles all imports and state.

## Dataset

The CERT Insider Threat Dataset r4.2 is publicly available on Kaggle:
https://www.kaggle.com/datasets/andrihjonior/cert-insider-threat-dataset-r4-2

Dataset is not uploaded to this repository due to size.
See `data/README.md` for access instructions.

## Key Output Files

| File | Description |
|---|---|
| `insider_threat_rf_model.pkl` | Production Random Forest model |
| `insider_threat_scaler.pkl` | StandardScaler fitted on training data |
| `feature_cols.json` | 15 feature columns in order |
| `results_summary.json` | Complete results record for reproducibility |
| `shap_bar.png` | Primary explainability output for publications |
| `roc_curve_comparison.png` | Model comparison visual |
