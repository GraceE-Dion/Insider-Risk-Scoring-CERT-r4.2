# Repository Structure

## Insider Risk Scoring - CERT r4.2

```
Insider-Risk-Scoring-CERT-r4.2/
│
│  ── Inherited from Insider Threat Detection pipeline (Scripts 01-18) ──
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
├── 12_pr_auc.py                     # Precision-Recall AUC comparison
├── 13_cost_matrix.py                # Cost matrix and economic threshold analysis
├── 14_user_id_verification.py       # User-ID split leakage verification
├── 15_sensitivity_test.py           # Adversarial sensitivity test on unique_devices
├── 16_velocity_features.py          # Velocity feature engineering (4 features)
├── 17_enhanced_model.py             # Enhanced model training (19 features)
├── 18_enhanced_shap.py              # SHAP explainability for enhanced model
│
│  ── Risk Scoring Extension (Scripts 19-26) ──
│
├── 19_risk_score_distribution.py    # Continuous risk scores for all 1,000 users
├── 20_risk_tier_classification.py   # Tier segmentation (Critical/High/Medium/Low) + lift
├── 21_shap_tier_analysis.py         # Per-tier SHAP drivers + explainability gap
├── 22_risk_user_report.py           # Ranked risk report with top 3 drivers per user
├── 23_score_calibration.py          # Isotonic regression and Platt scaling calibration
├── 24_temporal_trajectory.py        # 17-month rolling risk score trajectories
├── 25_risk_velocity.py              # Monthly velocity alerting (score change)
├── 26_feature_stability.py          # CV rank stability + policy-to-feature mapping
│
├── master_training_script.py        # End-to-end pipeline (single script reproduction)
├── INHERITED_SCRIPTS_NOTE.md        # Documents scripts inherited from threat detection repo
│
├── notebooks/
│   └── Insider_Risk_Scoring_CERT_r4.2.ipynb   # Kaggle notebook export
│
├── models/
│   ├── insider_threat_rf_model.pkl  # Base Random Forest model (from threat detection)
│   ├── insider_threat_scaler.pkl    # Fitted StandardScaler
│   └── README.md                    # Model access instructions
│
├── images/
│   ├── Risk_Score_Distribution.png
│   ├── Risk_Score_Density.png
│   ├── Risk_Tier_User_Counts.png
│   ├── Risk_Tier_Malicious_Rate.png
│   ├── Risk_Tier_Lift.png
│   ├── SHAP_Tier_Drivers.png
│   ├── SHAP_Explainability_Gap.png
│   ├── Risk_Score_Calibration_Curve.png
│   ├── Risk_Score_Distribution_Calibrated.png
│   ├── Risk_Trajectory_Malicious.png
│   ├── Risk_Trajectory_Benign.png
│   ├── Risk_Velocity_Malicious.png
│   ├── Risk_Velocity_Distribution.png
│   └── Feature_Stability.png
│
├── outputs/
│   └── Insider_Risk_Report.csv      # All 1,000 users ranked by risk score
│
├── data/
│   └── README.md                    # Dataset access instructions (not uploaded)
│
├── feature_cols.json                # Ordered feature column list (19 features)
├── results_summary.json             # Full project results record
├── README.md                        # Project documentation
├── STRUCTURE.md                     # This file
├── INHERITED_SCRIPTS_NOTE.md        # Scripts inherited from threat detection repo
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Ignored files
```

## Script Execution Order

**Scripts 01-18** are inherited from the Insider Threat Detection pipeline:
https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2

Run them sequentially first to restore the enhanced 19-feature Random Forest model
(rf_v2), scaler (scaler_v2), and all feature engineering outputs into memory.

**Scripts 19-26** implement the continuous risk scoring extension and must be
run after scripts 01-18 are complete.

Use `master_training_script.py` to reproduce the full pipeline end-to-end
from a single execution.

## Key Output Files

| File | Description |
|---|---|
| `Insider_Risk_Report.csv` | All 1,000 users ranked by risk score with tier and top 3 drivers |
| `Risk_Score_Calibration_Curve.png` | Reliability diagram for audit requirements |
| `Risk_Trajectory_Malicious.png` | 17-month behavioral ramp-up visualization |
| `Risk_Tier_Lift.png` | Executive ROI metric - Critical tier 14.1x lift |
| `Feature_Stability.png` | CMMC maturity: all 19 features HIGH stability |

## Relationship to Insider Threat Detection Project

This repository extends:
https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2

The threat detection project performs binary classification (malicious vs benign).
This project transforms those binary outputs into a continuous risk scoring system
aligned with the ISO 27001 risk-based approach and NIST SP 800-37 continuous
monitoring requirements. Together the two projects represent a complete
human-factor behavioral analytics pipeline spanning detection and quantification.
