# =============================================================================
# 11_save_outputs.py
# Insider Threat Detection — CERT r4.2
# Step 11: Save trained model, scaler, feature columns, and results summary
#
# Saves:
#   insider_threat_rf_model.pkl  — trained Random Forest model
#   insider_threat_scaler.pkl    — fitted StandardScaler
#   feature_cols.json            — ordered feature column list
#   results_summary.json         — full project results record
# =============================================================================

import joblib
import json

print("=" * 65)
print("Step 11: Save Model, Scaler, and Results")
print("=" * 65)

# ── Save model and scaler ─────────────────────────────────────────────────────
joblib.dump(rf_model, "insider_threat_rf_model.pkl")
joblib.dump(scaler,   "insider_threat_scaler.pkl")
print("Saved: insider_threat_rf_model.pkl")
print("Saved: insider_threat_scaler.pkl")

# ── Save feature columns ──────────────────────────────────────────────────────
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
print("Saved: feature_cols.json")

# ── Save full results summary ─────────────────────────────────────────────────
summary = {
    "project": "Insider Threat Detection — CERT r4.2",
    "author": "Grace Egbedion",
    "niw_framing": (
        "Human-factor risk analytics for cybersecurity governance "
        "aligned with NIST SP 800-37, CMMC, ISO 27001"
    ),
    "dataset": {
        "name": "CERT Insider Threat Dataset r4.2",
        "source": "Kaggle — andrihjonior/cert-insider-threat-dataset-r4-2",
        "total_users": 1000,
        "malicious_users": 70,
        "class_imbalance": "93% benign / 7% malicious"
    },
    "features": {
        "total_features": len(feature_cols),
        "feature_list": feature_cols,
        "sources": [
            "logon activity (4 features)",
            "device activity (2 features)",
            "email activity (4 features)",
            "psychometric Big Five (5 features)"
        ]
    },
    "preprocessing": {
        "missing_value_strategy": "zero-fill (absence of activity)",
        "train_test_split": "80/20 stratified",
        "scaling": "StandardScaler (fit on train only)",
        "class_balancing": "SMOTE on training set only"
    },
    "model_results": {
        "Random Forest": {
            "recall_malicious": 1.00,
            "precision_malicious": 0.78,
            "f1_malicious": 0.88,
            "roc_auc": 0.9939,
            "cv_auc_mean": 0.9967,
            "cv_auc_std": 0.0023,
            "false_negatives": 0,
            "false_positives": 4
        },
        "XGBoost": {
            "recall_malicious": 0.93,
            "precision_malicious": 0.76,
            "f1_malicious": 0.84,
            "roc_auc": 0.9965
        }
    },
    "selected_model": "Random Forest",
    "selection_rationale": (
        "Perfect recall (1.00) — zero missed insiders in test set. "
        "In insider threat detection, false negatives are operationally "
        "unacceptable. Random Forest achieves perfect detection at the "
        "cost of 4 false positives per 200 users evaluated."
    ),
    "top_shap_features": [
        {"feature": "unique_devices",    "shap_impact": 0.1453},
        {"feature": "logoff_count",      "shap_impact": 0.1254},
        {"feature": "device_count",      "shap_impact": 0.1155},
        {"feature": "logon_count",       "shap_impact": 0.0391},
        {"feature": "after_hours_logon", "shap_impact": 0.0336}
    ],
    "key_finding": (
        "Behavioral signals (device access, session patterns, after-hours "
        "activity) are 5-10x more predictive than psychometric personality "
        "traits for insider threat detection. This empirically validates "
        "the human-factor behavioral analytics approach in the "
        "SECURE-EXEC governance framework."
    ),
    "operational_threshold": {
        "recommended": 0.50,
        "recall_at_threshold": 1.00,
        "precision_at_threshold": 0.78,
        "f1_at_threshold": 0.88
    }
}

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: results_summary.json")

print("\nAll outputs saved successfully.")
print("\nFiles for GitHub upload:")
print("  models/  : insider_threat_rf_model.pkl, insider_threat_scaler.pkl")
print("  outputs/ : shap_summary.png, shap_bar.png, shap_force.png,")
print("             threshold_curve.png, roc_curve_comparison.png,")
print("             confusion_matrix_random_forest.png,")
print("             confusion_matrix_xgboost.png")
print("  root/    : feature_cols.json, results_summary.json")

print("\n" + "=" * 65)
print("STATUS: All outputs saved. Project pipeline complete.")
print("=" * 65)
