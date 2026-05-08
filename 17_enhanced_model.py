# =============================================================================
# 17_enhanced_model.py
# Insider Threat Detection — CERT r4.2
# Step 17: Enhanced Model Training (15 + 4 Velocity Features = 19 Features)
#
# Retrains Random Forest with velocity features added to the original
# 15-feature set. Evaluates improvement across all metrics.
#
# Enhanced results vs baseline:
#   ROC-AUC:  0.9985 (was 0.9939, +0.0046)
#   PR-AUC:   0.9841 (was 0.9167, +0.0674)
#   CV AUC:   0.9991 +/- 0.0013 (was 0.9967 +/- 0.0023, tighter)
#   Recall:   1.00   (unchanged — still zero missed insiders)
#   FN:       0      (unchanged)
#   FP:       4      (unchanged)
#
# Outputs: Confusion_Matrix_Enhanced_Model.png
#          Features_Importances_Enhanced_Model.png
# =============================================================================

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 17: Enhanced Model Training (19 Features)")
print("=" * 55)

velocity_cols = ['after_hours_ratio', 'unique_devices_per_day',
                 'logon_burst_ratio', 'email_burst_ratio']

# ── Merge velocity features ───────────────────────────────────────────────────
merged_v2 = merged.merge(velocity_features, on='user', how='left')
merged_v2[velocity_cols] = merged_v2[velocity_cols].fillna(0)

feature_cols_v2 = [c for c in merged_v2.columns
                   if c not in ['user', 'label']]

print(f"\nOriginal feature count: {len(feature_cols)}")
print(f"Enhanced feature count: {len(feature_cols_v2)}")
print(f"\nNew velocity features added:")
for f in velocity_cols:
    print(f"  + {f}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
X_v2 = merged_v2[feature_cols_v2].values
y_v2 = merged_v2['label'].values

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    X_v2, y_v2, test_size=0.20, random_state=42, stratify=y_v2)

scaler_v2             = StandardScaler()
X_train_v2_scaled     = scaler_v2.fit_transform(X_train_v2)
X_test_v2_scaled      = scaler_v2.transform(X_test_v2)

smote_v2              = SMOTE(random_state=42)
X_train_v2_sm, y_train_v2_sm = smote_v2.fit_resample(
    X_train_v2_scaled, y_train_v2)

print(f"\nTrain shape after SMOTE: {X_train_v2_sm.shape}")

# ── Train ─────────────────────────────────────────────────────────────────────
rf_v2 = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf_v2.fit(X_train_v2_sm, y_train_v2_sm)

y_pred_v2  = rf_v2.predict(X_test_v2_scaled)
y_proba_v2 = rf_v2.predict_proba(X_test_v2_scaled)[:, 1]

auc_v2   = roc_auc_score(y_test_v2, y_proba_v2)
prauc_v2 = average_precision_score(y_test_v2, y_proba_v2)
fn_v2    = ((y_test_v2 == 1) & (y_pred_v2 == 0)).sum()
fp_v2    = ((y_test_v2 == 0) & (y_pred_v2 == 1)).sum()

print(f"\n{'='*55}")
print("Enhanced Model Results (19 features):")
print(f"{'='*55}")
print(classification_report(y_test_v2, y_pred_v2,
                             target_names=['Benign', 'Malicious']))
print(f"ROC-AUC:         {auc_v2:.4f}")
print(f"PR-AUC:          {prauc_v2:.4f}")
print(f"False Negatives: {fn_v2}")
print(f"False Positives: {fp_v2}")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv_v2       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_v2 = cross_val_score(rf_v2, X_v2, y_v2,
                                cv=cv_v2, scoring='roc_auc')
print(f"\n5-Fold CV AUC: {cv_scores_v2.mean():.4f} "
      f"(+/- {cv_scores_v2.std():.4f})")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_v2   = confusion_matrix(y_test_v2, y_pred_v2)
disp_v2 = ConfusionMatrixDisplay(cm_v2,
                                  display_labels=['Benign', 'Malicious'])
disp_v2.plot(cmap='Blues')
plt.title("Enhanced Model (19 Features) — Confusion Matrix")
plt.tight_layout()
plt.savefig("Confusion_Matrix_Enhanced_Model.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Confusion_Matrix_Enhanced_Model.png")

# ── Feature importance ────────────────────────────────────────────────────────
importances_v2 = rf_v2.feature_importances_
indices_v2     = np.argsort(importances_v2)[::-1]
colors         = ['#E67E22' if feature_cols_v2[i] in velocity_cols
                  else '#2980B9' for i in indices_v2]

print("\nAll 19 features ranked by importance:")
for rank, idx in enumerate(indices_v2, 1):
    marker = " <-- NEW" if feature_cols_v2[idx] in velocity_cols else ""
    print(f"  {rank:>2}. {feature_cols_v2[idx]:<35} "
          f"{importances_v2[idx]:.4f}{marker}")

plt.figure(figsize=(11, 6))
plt.title("Feature Importances — Enhanced Model (19 Features)\n"
          "Blue = Original | Orange = Velocity Features")
plt.bar(range(len(indices_v2)),
        importances_v2[indices_v2],
        color=colors, align='center')
plt.xticks(range(len(indices_v2)),
           [feature_cols_v2[i] for i in indices_v2],
           rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig("Features_Importances_Enhanced_Model.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Features_Importances_Enhanced_Model.png")

print("=" * 55)
print("STATUS: Enhanced model complete. Proceed to Step 18.")
print("=" * 55)
