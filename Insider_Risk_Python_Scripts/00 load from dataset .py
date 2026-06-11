# =============================================================================
# 00_load_from_dataset.py
# Insider Risk Scoring — CERT r4.2
# Step 0: Load saved model artefacts from Kaggle dataset
#
# This script replaces running Steps 01-18 from the Insider Threat Detection
# pipeline. It loads the pre-trained enhanced Random Forest model (rf_v2),
# fitted scaler (scaler_v2), and feature column list (feature_cols_v2)
# directly from the saved Kaggle dataset, then reconstructs all in-memory
# variables needed by Steps 19-29.
#
# Required Kaggle dataset: insider-threat-models
#   Add it to your notebook via: Data -> Add Data -> Your Datasets
#   Dataset path: /kaggle/input/insider-threat-models/
#
# Required CERT dataset: andrihjonior/cert-insider-threat-dataset-r4-2
#   (same as threat detection notebook)
#
# After running this script, all variables are in memory and Steps 19-29
# can run without any retraining.
#
# Author: Grace Egbedion
# =============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 60)
print("INSIDER RISK SCORING — CERT r4.2")
print("Step 0: Load Pre-trained Model from Kaggle Dataset")
print("=" * 60)

# ── Dataset paths ─────────────────────────────────────────────────────────────
MODEL_PATH = '/kaggle/input/insider-threat-models/'
BASE       = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/r4.2/'
ANSWERS    = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/answers/insiders.csv'

# ── Verify model files exist ──────────────────────────────────────────────────
required_model_files = [
    'rf_v2_enhanced_model.pkl',
    'scaler_v2.pkl',
    'feature_cols_v2.json',
]
print("\nVerifying model artefacts:")
for fname in required_model_files:
    fpath  = MODEL_PATH + fname
    exists = os.path.exists(fpath)
    print(f"  [{'FOUND  ' if exists else 'MISSING'}] {fpath}")

# ── Load model artefacts ──────────────────────────────────────────────────────
print("\nLoading model artefacts...")
rf_v2         = joblib.load(MODEL_PATH + 'rf_v2_enhanced_model.pkl')
scaler_v2     = joblib.load(MODEL_PATH + 'scaler_v2.pkl')

with open(MODEL_PATH + 'feature_cols_v2.json') as f:
    feature_cols_v2 = json.load(f)

velocity_cols = ['after_hours_ratio', 'unique_devices_per_day',
                 'logon_burst_ratio', 'email_burst_ratio']

print(f"rf_v2:           {rf_v2}")
print(f"feature_cols_v2: {len(feature_cols_v2)} features")
print(f"velocity_cols:   {velocity_cols}")

# ── Reload raw data ───────────────────────────────────────────────────────────
print("\nLoading raw data files...")
logon        = pd.read_csv(BASE + 'logon.csv')
device       = pd.read_csv(BASE + 'device.csv')
psychometric = pd.read_csv(BASE + 'psychometric.csv')
insiders     = pd.read_csv(ANSWERS)
email        = pd.read_csv(BASE + 'email.csv')

print(f"  Logon:        {logon.shape}")
print(f"  Device:       {device.shape}")
print(f"  Email:        {email.shape}")
print(f"  Psychometric: {psychometric.shape}")

# ── Rebuild feature tables (same logic as Steps 03-04) ───────────────────────
print("\nRebuilding feature tables...")
logon['date']           = pd.to_datetime(logon['date'])
logon['hour']           = logon['date'].dt.hour
logon['is_after_hours'] = ((logon['hour'] < 7) | (logon['hour'] > 18)).astype(int)

logon_features = logon.groupby('user').agg(
    logon_count      =('activity', lambda x: (x=='Logon').sum()),
    logoff_count     =('activity', lambda x: (x=='Logoff').sum()),
    after_hours_logon=('is_after_hours','sum'),
    unique_pcs       =('pc','nunique')
).reset_index()

device_features = device.groupby('user').agg(
    device_count  =('activity','count'),
    unique_devices=('pc','nunique')
).reset_index()

email_features = email.groupby('user').agg(
    email_sent       =('id','count'),
    avg_email_size   =('size','mean'),
    total_attachments=('attachments','sum'),
    unique_recipients=('to','nunique')
).reset_index()

psych_features = psychometric[['user_id','O','C','E','A','N']].copy()
psych_features.columns = ['user','openness','conscientiousness',
                           'extraversion','agreeableness','neuroticism']

# Merge baseline
merged = psych_features.copy()
merged = merged.merge(logon_features,  on='user', how='left')
merged = merged.merge(device_features, on='user', how='left')
merged = merged.merge(email_features,  on='user', how='left')

insider_users  = set(insiders['user'].unique())
merged['label'] = merged['user'].apply(
    lambda u: 1 if u in insider_users else 0)

feature_cols         = [c for c in merged.columns if c not in ['user','label']]
merged[feature_cols] = merged[feature_cols].fillna(0)

# ── Rebuild velocity features (same logic as Step 16) ────────────────────────
print("Rebuilding velocity features...")
user_active_days = logon.groupby('user')['date'].apply(
    lambda x: x.dt.date.nunique()).reset_index()
user_active_days.columns = ['user','active_days']

ah_ratio = logon.groupby('user').agg(
    total_logons    =('activity','count'),
    total_ah_logons =('is_after_hours','sum')
).reset_index()
ah_ratio['after_hours_ratio'] = (
    ah_ratio['total_ah_logons'] / ah_ratio['total_logons']).round(4)

device_days = device.groupby('user').agg(
    unique_devs=('pc','nunique')).reset_index()
device_days = device_days.merge(user_active_days, on='user', how='left')
device_days['unique_devices_per_day'] = (
    device_days['unique_devs'] / device_days['active_days']).round(4)

logon['date_only'] = logon['date'].dt.date
daily_logons       = logon.groupby(
    ['user','date_only']).size().reset_index(name='dc')
lbr_stats          = daily_logons.groupby('user').agg(
    avg_dl=('dc','mean'), peak_dl=('dc','max')).reset_index()
lbr_stats['logon_burst_ratio'] = (
    lbr_stats['peak_dl'] / lbr_stats['avg_dl']).round(4)

email['date_only'] = pd.to_datetime(email['date']).dt.date
daily_emails       = email.groupby(
    ['user','date_only']).size().reset_index(name='ec')
ebr_stats          = daily_emails.groupby('user').agg(
    avg_de=('ec','mean'), peak_de=('ec','max')).reset_index()
ebr_stats['email_burst_ratio'] = (
    ebr_stats['peak_de'] / ebr_stats['avg_de']).round(4)

velocity_features = ah_ratio[['user','after_hours_ratio']].copy()
velocity_features = velocity_features.merge(
    device_days[['user','unique_devices_per_day']], on='user', how='left')
velocity_features = velocity_features.merge(
    lbr_stats[['user','logon_burst_ratio']], on='user', how='left')
velocity_features = velocity_features.merge(
    ebr_stats[['user','email_burst_ratio']], on='user', how='left')

# ── Rebuild merged_v2 ─────────────────────────────────────────────────────────
merged_v2                = merged.merge(velocity_features, on='user', how='left')
merged_v2[velocity_cols] = merged_v2[velocity_cols].fillna(0)

# ── Reconstruct train/test split (same random_state as training) ──────────────
print("Reconstructing train/test split...")
X_v2 = merged_v2[feature_cols_v2].values
y_v2 = merged_v2['label'].values

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    X_v2, y_v2, test_size=0.20, random_state=42, stratify=y_v2)

X_train_v2_scaled = scaler_v2.transform(X_train_v2)
X_test_v2_scaled  = scaler_v2.transform(X_test_v2)

# ── Reconstruct predictions ───────────────────────────────────────────────────
y_pred_v2  = rf_v2.predict(X_test_v2_scaled)
y_proba_v2 = rf_v2.predict_proba(X_test_v2_scaled)[:, 1]

auc_v2   = roc_auc_score(y_test_v2, y_proba_v2)
prauc_v2 = average_precision_score(y_test_v2, y_proba_v2)
fn_v2    = ((y_test_v2==1) & (y_pred_v2==0)).sum()
fp_v2    = ((y_test_v2==0) & (y_pred_v2==1)).sum()

# ── Reconstruct SHAP values ───────────────────────────────────────────────────
print("Reconstructing SHAP values (may take ~30 seconds)...")
explainer_v2     = shap.TreeExplainer(rf_v2)
shap_values_v2   = explainer_v2.shap_values(X_test_v2_scaled)
shap_vals_v2_mal = shap_values_v2[:, :, 1]

# ── Verification ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION — all variables in memory:")
print(f"  merged_v2 shape:          {merged_v2.shape}")
print(f"  feature_cols_v2 count:    {len(feature_cols_v2)}")
print(f"  X_test_v2_scaled shape:   {X_test_v2_scaled.shape}")
print(f"  y_test_v2 shape:          {y_test_v2.shape}")
print(f"  shap_vals_v2_mal shape:   {shap_vals_v2_mal.shape}")
print(f"  ROC-AUC (reconstructed):  {auc_v2:.4f}")
print(f"  PR-AUC  (reconstructed):  {prauc_v2:.4f}")
print(f"  False Negatives:          {fn_v2}")
print(f"  False Positives:          {fp_v2}")
print("=" * 60)
print("\nAll variables ready. Proceed to Step 19.")
print("=" * 60)
