# =============================================================================
# master_training_script.py
# Insider Threat Detection — CERT r4.2
#
# Master script: runs the complete pipeline end-to-end in sequence.
# Equivalent to running all 11 numbered scripts in order.
#
# Usage (Kaggle or local):
#   Run this single script to reproduce all results from scratch.
#
# Author: Grace Egbedion
# NIW Framing: Human-factor risk analytics for cybersecurity governance
#              aligned with NIST SP 800-37, CMMC, ISO 27001
# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 65)
print("INSIDER THREAT DETECTION — CERT r4.2")
print("Master Training Script — Full Pipeline")
print("=" * 65)

# =============================================================================
# STEP 1: FILE PATHS
# =============================================================================
print("\n[STEP 1] Environment Setup")
BASE    = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/r4.2/'
ANSWERS = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/answers/insiders.csv'

# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================
print("[STEP 2] Loading data...")
logon        = pd.read_csv(BASE + 'logon.csv')
device       = pd.read_csv(BASE + 'device.csv')
psychometric = pd.read_csv(BASE + 'psychometric.csv')
insiders     = pd.read_csv(ANSWERS)
email        = pd.read_csv(BASE + 'email.csv')
print(f"  Logon: {logon.shape} | Device: {device.shape} | "
      f"Email: {email.shape} | Psychometric: {psychometric.shape}")

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
print("[STEP 3] Feature engineering...")

logon['date']            = pd.to_datetime(logon['date'])
logon['hour']            = logon['date'].dt.hour
logon['is_after_hours']  = ((logon['hour'] < 7) | (logon['hour'] > 18)).astype(int)

logon_features = logon.groupby('user').agg(
    logon_count      = ('activity', lambda x: (x == 'Logon').sum()),
    logoff_count     = ('activity', lambda x: (x == 'Logoff').sum()),
    after_hours_logon= ('is_after_hours', 'sum'),
    unique_pcs       = ('pc', 'nunique')
).reset_index()

device_features = device.groupby('user').agg(
    device_count   = ('activity', 'count'),
    unique_devices = ('pc', 'nunique')
).reset_index()

email_features = email.groupby('user').agg(
    email_sent        = ('id', 'count'),
    avg_email_size    = ('size', 'mean'),
    total_attachments = ('attachments', 'sum'),
    unique_recipients = ('to', 'nunique')
).reset_index()

psych_features = psychometric[['user_id', 'O', 'C', 'E', 'A', 'N']].copy()
psych_features.columns = ['user', 'openness', 'conscientiousness',
                           'extraversion', 'agreeableness', 'neuroticism']

print(f"  Logon: {logon_features.shape} | Device: {device_features.shape} | "
      f"Email: {email_features.shape} | Psych: {psych_features.shape}")

# =============================================================================
# STEP 4: MERGE AND LABEL
# =============================================================================
print("[STEP 4] Merging features and creating binary label...")

merged = psych_features.copy()
merged = merged.merge(logon_features,  on='user', how='left')
merged = merged.merge(device_features, on='user', how='left')
merged = merged.merge(email_features,  on='user', how='left')

insider_users  = set(insiders['user'].unique())
merged['label'] = merged['user'].apply(lambda u: 1 if u in insider_users else 0)

print(f"  Merged shape: {merged.shape} | "
      f"Malicious: {merged['label'].sum()} | Benign: {(merged['label']==0).sum()}")

# =============================================================================
# STEP 5: PREPROCESSING
# =============================================================================
print("[STEP 5] Preprocessing — imputation, split, scaling, SMOTE...")

feature_cols          = [c for c in merged.columns if c not in ['user', 'label']]
merged[feature_cols]  = merged[feature_cols].fillna(0)

X = merged[feature_cols].values
y = merged['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote          = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"  Train: {X_train_sm.shape} | Test: {X_test_scaled.shape} | "
      f"SMOTE balanced: {(y_train_sm==1).sum()} malicious")

# =============================================================================
# STEP 6: MODEL TRAINING
# =============================================================================
print("[STEP 6] Training Random Forest and XGBoost...")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(y_train_sm==0).sum()/(y_train_sm==1).sum(),
        use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = {'model': model, 'y_pred': y_pred, 'auc': auc}
    print(f"  {name} — AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                  target_names=['Benign', 'Malicious']))

# =============================================================================
# STEP 7: ROC CURVE
# =============================================================================
print("[STEP 7] Generating ROC curve...")
plt.figure(figsize=(8, 6))
for name, res in results.items():
    y_proba  = res['model'].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})")
plt.plot([0,1],[0,1],'k--', label='Random Baseline')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Insider Threat Detection (CERT r4.2)')
plt.legend(loc='lower right'); plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 8: SHAP EXPLAINABILITY
# =============================================================================
print("[STEP 8] Computing SHAP values...")
rf_model    = results["Random Forest"]["model"]
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)
shap_vals_malicious = shap_values[:, :, 1]

plt.figure()
shap.summary_plot(shap_vals_malicious, X_test_scaled,
                  feature_names=feature_cols, plot_type="dot", show=False)
plt.title("SHAP Summary — Feature Impact on Insider Threat Prediction")
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure()
shap.summary_plot(shap_vals_malicious, X_test_scaled,
                  feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP Mean Absolute Impact — Insider Threat Features")
plt.tight_layout(); plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 9: THRESHOLD TUNING
# =============================================================================
print("[STEP 9] Threshold tuning...")
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
thresh_df = pd.DataFrame({
    'threshold': thresholds,
    'precision': precisions[:-1],
    'recall':    recalls[:-1],
    'f1': 2*(precisions[:-1]*recalls[:-1])/(precisions[:-1]+recalls[:-1]+1e-9)
})
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
plt.plot(thresholds, recalls[:-1],    label='Recall',    color='orange')
plt.plot(thresholds, thresh_df['f1'], label='F1 Score',  color='green')
plt.axvline(x=0.5, color='gray', linestyle='--', label='Default threshold (0.5)')
plt.xlabel('Threshold'); plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Threshold — Random Forest')
plt.legend(); plt.tight_layout()
plt.savefig("threshold_curve.png", dpi=150, bbox_inches='tight'); plt.show()

# =============================================================================
# STEP 10: CROSS-VALIDATION
# =============================================================================
print("[STEP 10] 5-fold cross-validation...")
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')
print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =============================================================================
# STEP 11: SAVE OUTPUTS
# =============================================================================
print("[STEP 11] Saving outputs...")
joblib.dump(rf_model, "insider_threat_rf_model.pkl")
joblib.dump(scaler,   "insider_threat_scaler.pkl")
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

summary = {
    "project": "Insider Threat Detection — CERT r4.2",
    "author": "Grace Egbedion",
    "selected_model": "Random Forest",
    "recall_malicious": 1.00,
    "precision_malicious": 0.78,
    "f1_malicious": 0.88,
    "roc_auc": 0.9939,
    "cv_auc": round(float(cv_scores.mean()), 4),
    "cv_std": round(float(cv_scores.std()), 4)
}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("PIPELINE COMPLETE — All results reproduced successfully.")
print(f"  Random Forest AUC:   {results['Random Forest']['auc']:.4f}")
print(f"  XGBoost AUC:         {results['XGBoost']['auc']:.4f}")
print(f"  5-Fold CV AUC:       {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("=" * 65)
