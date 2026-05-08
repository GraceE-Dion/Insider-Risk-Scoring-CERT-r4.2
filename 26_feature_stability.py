# =============================================================================
# 26_feature_stability.py
# Insider Risk Scoring — CERT r4.2
# Step 26: Feature stability and policy-to-feature mapping
#
# Computes SHAP rank stability across 5 CV folds to verify that feature
# importance rankings are reproducible properties of the detection system,
# not artifacts of a particular data split.
#
# Also documents a policy-to-feature mapping showing which organizational
# policy changes (BYOD, remote work, DLP, etc.) would affect which features
# and whether model retraining would be required. This is a CMMC maturity
# requirement for ML-based security systems.
#
# Key results:
#   All 19 features: HIGH stability (std < 2.0 across all folds)
#   Top 6 features (unique_devices_per_day through logon_count): std = 0.00
#   5 of 6 policy scenarios require model retraining
#
# Output: Feature_Stability.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("=" * 55)
print("Step 26: Feature Stability and Policy-to-Feature Mapping")
print("=" * 55)

cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_ranks = []

print("\nComputing SHAP ranks across 5 folds (may take ~60 seconds)...")

for fold, (train_idx, test_idx) in enumerate(cv.split(X_all_scaled, y_all), 1):
    X_tr, X_te = X_all_scaled[train_idx], X_all_scaled[test_idx]
    y_tr       = y_all[train_idx]

    smote_f          = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote_f.fit_resample(X_tr, y_tr)

    rf_fold = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf_fold.fit(X_tr_sm, y_tr_sm)

    exp_fold  = shap.TreeExplainer(rf_fold)
    sv_fold   = exp_fold.shap_values(X_te)
    sv_mal    = sv_fold[:, :, 1]
    mean_shap = np.abs(sv_mal).mean(axis=0)
    ranks     = pd.Series(mean_shap, index=feature_cols_v2).rank(
        ascending=False)
    fold_ranks.append(ranks)
    print(f"  Fold {fold} done")

rank_df      = pd.DataFrame(fold_ranks)
mean_rank    = rank_df.mean()
std_rank     = rank_df.std()
stability_df = pd.DataFrame({
    'feature':   feature_cols_v2,
    'mean_rank': mean_rank.values,
    'std_rank':  std_rank.values
}).sort_values('mean_rank')

print(f"\nFeature Stability (lower std = more stable rank):")
print(f"{'Feature':<30} {'Mean Rank':<12} {'Std Dev':<12} {'Stability'}")
print("-" * 65)
for _, row in stability_df.iterrows():
    stability = ("HIGH"   if row['std_rank'] < 2 else
                 "MEDIUM" if row['std_rank'] < 4 else "LOW")
    print(f"{row['feature']:<30} {row['mean_rank']:<12.2f} "
          f"{row['std_rank']:<12.2f} {stability}")

# ── Policy-to-feature mapping ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Policy-to-Feature Mapping (CMMC Alignment)")
print("=" * 55)

policy_map = {
    'BYOD Policy Change':
        ['unique_devices','unique_devices_per_day','device_count'],
    'Remote Work / WFH Policy':
        ['after_hours_logon','after_hours_ratio','unique_pcs'],
    'Email DLP Policy':
        ['email_sent','total_attachments','unique_recipients',
         'avg_email_size','email_burst_ratio'],
    'Shared Workstation Policy':
        ['unique_pcs','logon_count','logoff_count','logon_burst_ratio'],
    'USB/Removable Media Policy':
        ['unique_devices','device_count','unique_devices_per_day'],
    'Flex Hours / Shift Work Policy':
        ['after_hours_logon','after_hours_ratio']
}

print(f"\n{'Policy Change':<30} {'Affected Features':<45} {'Retraining'}")
print("-" * 85)
for policy, features in policy_map.items():
    shap_impact = sum(
        stability_df[stability_df['feature']==f]['mean_rank'].values[0]
        if f in stability_df['feature'].values else 99
        for f in features)
    retrain = "YES — High Impact" if shap_impact < 30 else "MONITOR"
    print(f"{policy:<30} {', '.join(features[:2]) + '...':<45} {retrain}")

# ── Plot: Feature stability ───────────────────────────────────────────────────
top_stable = stability_df.head(10)
colors     = ['#2ECC71' if s < 2 else '#F39C12' if s < 4 else '#E74C3C'
              for s in top_stable['std_rank']]

plt.figure(figsize=(11, 6))
plt.barh(top_stable['feature'], top_stable['std_rank'],
         color=colors, edgecolor='gray')
plt.axvline(x=2, color='green',  linestyle='--', linewidth=1.5,
            label='High stability threshold (std < 2)')
plt.axvline(x=4, color='orange', linestyle='--', linewidth=1.5,
            label='Medium stability threshold (std < 4)')
plt.xlabel('Rank Standard Deviation Across 5 Folds', fontsize=13)
plt.ylabel('Feature', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Feature Rank Stability Across CV Folds\n'
          '(Green = High, Orange = Medium, Red = Low stability)',
          fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Feature_Stability.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: Feature_Stability.png")

print("\n" + "=" * 55)
print("STATUS: Feature stability complete. All Steps 19-26 done.")
print("=" * 55)
