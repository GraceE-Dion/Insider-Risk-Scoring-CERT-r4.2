# =============================================================================
# 04_merge_and_label.py
# Insider Threat Detection — CERT r4.2
# Step 4: Merge all feature tables and create binary target label
#
# Merges: psych_features + logon_features + device_features + email_features
# Creates binary label: 1 = malicious insider, 0 = benign
# Output: merged DataFrame (1000, 16) with 'label' column
#
# NIW Framing: Label creation operationalizes insider threat classification
#              aligned with NIST SP 800-37 continuous monitoring requirements
# =============================================================================

import pandas as pd

print("=" * 65)
print("Step 4: Merge Feature Tables and Create Binary Target Label")
print("=" * 65)

# ── Merge all feature tables on 'user' ────────────────────────────────────────
# Start with psych_features — guarantees all 1,000 users are represented
merged = psych_features.copy()

merged = merged.merge(logon_features,  on='user', how='left')
merged = merged.merge(device_features, on='user', how='left')
merged = merged.merge(email_features,  on='user', how='left')

print(f"Merged shape: {merged.shape}")
print(merged.head())

# ── Binary target label ───────────────────────────────────────────────────────
# Deduplicate insider users — 191 records but 70 unique malicious users
insider_users = set(insiders['user'].unique())

merged['label'] = merged['user'].apply(lambda u: 1 if u in insider_users else 0)

print(f"\nTotal users:          {len(merged)}")
print(f"Malicious (label=1):  {merged['label'].sum()}")
print(f"Benign    (label=0):  {(merged['label'] == 0).sum()}")
print(f"\nClass balance:")
print(merged['label'].value_counts(normalize=True).round(4))

# ── Note on label counts ──────────────────────────────────────────────────────
print("""
NOTE: 191 insider records in answers file vs 70 unique malicious users.
      191 = total insider activity events across scenarios.
      70  = distinct users with malicious activity (used for labeling).
      This is correct behavior — label is per user, not per event.
""")

print("=" * 65)
print("STATUS: Merge and labeling complete. Proceed to Step 5.")
print("=" * 65)
