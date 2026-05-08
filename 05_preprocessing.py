# =============================================================================
# 05_preprocessing.py
# Insider Threat Detection — CERT r4.2
# Step 5: Handle missing values, train/test split, feature scaling, SMOTE
#
# Pipeline:
#   1. Zero-fill missing values (absence of activity = 0 events)
#   2. Stratified 80/20 train/test split
#   3. StandardScaler (fit on train only)
#   4. SMOTE oversampling on training set only
#
# Output: X_train_sm, y_train_sm, X_test_scaled, y_test, feature_cols
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

print("=" * 65)
print("Step 5: Preprocessing — Imputation, Split, Scaling, SMOTE")
print("=" * 65)

# ── Feature columns ───────────────────────────────────────────────────────────
feature_cols = [c for c in merged.columns if c not in ['user', 'label']]

# ── 5.1: Handle missing values ────────────────────────────────────────────────
print("\nMissing values before imputation:")
print(merged.isnull().sum()[merged.isnull().sum() > 0])

# Zero-fill: NaN = no observed activity for that user — a meaningful signal
merged[feature_cols] = merged[feature_cols].fillna(0)

print(f"\nMissing values after imputation: "
      f"{merged[feature_cols].isnull().sum().sum()}")
print(f"Final feature matrix: {merged[feature_cols].shape}")

# ── 5.2: Train/test split ─────────────────────────────────────────────────────
X = merged[feature_cols].values
y = merged['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # preserve 7% malicious ratio in both splits
)

print(f"\nX_train: {X_train.shape} | y_train positives: {y_train.sum()}")
print(f"X_test:  {X_test.shape}  | y_test positives:  {y_test.sum()}")

# ── 5.3: Feature scaling ──────────────────────────────────────────────────────
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)   # fit only on train
X_test_scaled   = scaler.transform(X_test)        # transform only

print(f"\nScaling complete.")
print(f"X_train_scaled mean ≈ {X_train_scaled.mean():.4f} | "
      f"std ≈ {X_train_scaled.std():.4f}")

# ── 5.4: SMOTE oversampling ───────────────────────────────────────────────────
# Applied ONLY to training set — test set remains untouched
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"\nBefore SMOTE — Benign: {(y_train==0).sum()} | "
      f"Malicious: {(y_train==1).sum()}")
print(f"After  SMOTE — Benign: {(y_train_sm==0).sum()} | "
      f"Malicious: {(y_train_sm==1).sum()}")

print("""
NOTE: SMOTE applied to training data only.
      X_test_scaled preserved in its original form for realistic evaluation.
      Zero-fill used for missing device activity (absence = no events recorded).
""")

print("=" * 65)
print("STATUS: Preprocessing complete. Proceed to Step 6.")
print("=" * 65)
