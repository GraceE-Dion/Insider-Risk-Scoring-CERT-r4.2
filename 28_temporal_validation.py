# =============================================================================
# 28_temporal_validation.py
# Insider Risk Scoring — CERT r4.2
# Step 28: Temporal split validation for risk scoring
#
# Addresses peer review requirement for external/temporal validation.
# Tests whether the risk scoring framework generalizes across temporal
# partitions by retraining on the first 80% of users by activity date
# and evaluating on the remaining 20%.
#
# Key results:
#   Split boundary: 2010-01-04
#   Train: 806 users (58 malicious, 7.2%)
#   Test:  194 users (12 malicious, 6.2%)
#   Temporal recall:   1.0000 (matches random split 1.0000)
#   Temporal AUC:      0.9986 (matches random split 0.9985)
#   Temporal PR-AUC:   0.9812 (matches random split 0.9841)
#   False Negatives:   0
#
#   Conclusion: The risk scoring pipeline generalizes across temporal
#   partitions. The framework is not dependent on the composition of
#   a particular random split.
#
# Outputs: Temporal_Validation_Risk.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (recall_score, precision_score,
                             f1_score, roc_auc_score,
                             average_precision_score,
                             classification_report)
from imblearn.over_sampling import SMOTE

print("=" * 55)
print("Step 28: Temporal Validation — Risk Scoring")
print("=" * 55)

# ── Build temporal split ──────────────────────────────────────────────────────
print("\nBuilding temporal split by user first activity date...")
logon['date']     = pd.to_datetime(logon['date'])
user_first_date   = logon.groupby('user')['date'].min().reset_index()
user_first_date.columns = ['user', 'first_date']

merged_temporal   = merged_v2.merge(user_first_date, on='user', how='left')
merged_temporal   = merged_temporal.sort_values(
    'first_date').reset_index(drop=True)

split_idx  = int(len(merged_temporal) * 0.80)
split_date = merged_temporal.iloc[split_idx]['first_date']
train_mask = merged_temporal['first_date'] <= split_date
test_mask  = merged_temporal['first_date'] >  split_date

X_temp_tr  = merged_temporal[feature_cols_v2].values[train_mask]
y_temp_tr  = merged_temporal['label'].values[train_mask]
X_temp_te  = merged_temporal[feature_cols_v2].values[test_mask]
y_temp_te  = merged_temporal['label'].values[test_mask]

print(f"Temporal split boundary: {split_date.date()}")
print(f"Train: {train_mask.sum()} users | "
      f"Malicious: {y_temp_tr.sum()} ({y_temp_tr.mean()*100:.1f}%)")
print(f"Test:  {test_mask.sum()} users  | "
      f"Malicious: {y_temp_te.sum()} ({y_temp_te.mean()*100:.1f}%)")

# ── Scale and SMOTE ───────────────────────────────────────────────────────────
scaler_temp          = StandardScaler()
X_temp_tr_sc         = scaler_temp.fit_transform(X_temp_tr)
X_temp_te_sc         = scaler_temp.transform(X_temp_te)

smote_temp           = SMOTE(random_state=42)
X_temp_sm, y_temp_sm = smote_temp.fit_resample(X_temp_tr_sc, y_temp_tr)
print(f"\nAfter SMOTE — Train: {X_temp_sm.shape}")

# ── Train model ───────────────────────────────────────────────────────────────
rf_temp = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf_temp.fit(X_temp_sm, y_temp_sm)

y_pred_temp  = rf_temp.predict(X_temp_te_sc)
y_proba_temp = rf_temp.predict_proba(X_temp_te_sc)[:, 1]

print(f"\nTemporal Split Classification Report:")
print(classification_report(y_temp_te, y_pred_temp,
                            target_names=['Benign','Malicious'],
                            zero_division=0))

auc_temp   = roc_auc_score(y_temp_te, y_proba_temp)
prauc_temp = average_precision_score(y_temp_te, y_proba_temp)
fn_temp    = ((y_temp_te==1) & (y_pred_temp==0)).sum()
fp_temp    = ((y_temp_te==0) & (y_pred_temp==1)).sum()
rec_temp   = recall_score(y_temp_te, y_pred_temp, zero_division=0)
prec_temp  = precision_score(y_temp_te, y_pred_temp, zero_division=0)
f1_temp    = f1_score(y_temp_te, y_pred_temp, zero_division=0)

print(f"ROC-AUC:         {auc_temp:.4f}")
print(f"PR-AUC:          {prauc_temp:.4f}")
print(f"False Negatives: {fn_temp}")
print(f"False Positives: {fp_temp}")

# ── Comparison table ──────────────────────────────────────────────────────────
print(f"\n{'Metric':<20} {'Random Split':<18} {'Temporal Split'}")
print("-" * 55)
comparisons = [
    ("Recall",    "1.0000",  f"{rec_temp:.4f}"),
    ("Precision", "0.7778",  f"{prec_temp:.4f}"),
    ("F1",        "0.8750",  f"{f1_temp:.4f}"),
    ("ROC-AUC",   "0.9985",  f"{auc_temp:.4f}"),
    ("PR-AUC",    "0.9841",  f"{prauc_temp:.4f}"),
    ("False Neg", "0",       str(fn_temp)),
    ("False Pos", "4",       str(fp_temp)),
]
for row in comparisons:
    print(f"  {row[0]:<18} {row[1]:<18} {row[2]}")

# ── Risk score distribution under temporal split ──────────────────────────────
# Score all temporal test users using temporal model
risk_temp_scores = y_proba_temp
tier_temp        = [assign_tier(s) for s in risk_temp_scores]

print(f"\nRisk tier distribution (temporal test set):")
for tier in ['Critical','High','Medium','Low']:
    mask     = [t == tier for t in tier_temp]
    n        = sum(mask)
    n_mal    = y_temp_te[mask].sum() if n > 0 else 0
    mal_rate = n_mal/n*100 if n > 0 else 0
    print(f"  {tier:<12}: {n} users, {n_mal} malicious ({mal_rate:.1f}%)")

# ── Plot comparison ───────────────────────────────────────────────────────────
metrics   = ['Recall','Precision','F1','ROC-AUC','PR-AUC']
rand_vals = [1.0000, 0.7778, 0.8750, 0.9985, 0.9841]
temp_vals = [rec_temp, prec_temp, f1_temp, auc_temp, prauc_temp]

x = np.arange(len(metrics))
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, rand_vals, 0.4, label='Random Split',
        color='steelblue', alpha=0.85)
plt.bar(x + 0.2, temp_vals, 0.4, label='Temporal Split',
        color='crimson', alpha=0.85)
for i, (r, t) in enumerate(zip(rand_vals, temp_vals)):
    plt.text(i-0.2, r+0.005, f'{r:.3f}', ha='center', fontsize=10)
    plt.text(i+0.2, t+0.005, f'{t:.3f}', ha='center', fontsize=10)
plt.xticks(x, metrics, fontsize=12)
plt.yticks(fontsize=11)
plt.ylim(0.75, 1.05)
plt.ylabel('Score', fontsize=13)
plt.xlabel('Metric', fontsize=13)
plt.title('Temporal Split vs Random Split Validation\n'
          '(Insider Risk Scoring)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Temporal_Validation_Risk.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: Temporal_Validation_Risk.png")

print("\n" + "=" * 55)
print("STATUS: Temporal validation complete. Proceed to Step 29.")
print("=" * 55)
