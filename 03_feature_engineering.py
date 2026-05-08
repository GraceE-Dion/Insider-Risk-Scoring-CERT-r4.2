# =============================================================================
# 03_feature_engineering.py
# Insider Threat Detection — CERT r4.2
# Step 3: Aggregate activity data into per-user behavioral features
#
# Produces:
#   logon_features   — (1000, 5)  logon/logoff counts, after-hours, unique PCs
#   device_features  — (265, 3)   device count, unique devices
#   email_features   — (1000, 5)  email sent, avg size, attachments, recipients
#   psych_features   — (1000, 6)  Big Five personality scores
#
# NIW Framing: Behavioral feature extraction operationalizes human-factor
#              risk analytics — the core of SECURE-EXEC™ governance methodology
# =============================================================================

import pandas as pd
import numpy as np

print("=" * 65)
print("Step 3: Feature Engineering — Aggregate Activity Per User")
print("=" * 65)

# ── Assumes data loaded from 02_load_data.py ──────────────────────────────────
# In Kaggle, run cells sequentially so logon, device, email,
# psychometric, insiders are already in memory.

# ── Logon features ────────────────────────────────────────────────────────────
logon['date']           = pd.to_datetime(logon['date'])
logon['hour']           = logon['date'].dt.hour
logon['is_after_hours'] = ((logon['hour'] < 7) | (logon['hour'] > 18)).astype(int)

logon_features = logon.groupby('user').agg(
    logon_count      = ('activity', lambda x: (x == 'Logon').sum()),
    logoff_count     = ('activity', lambda x: (x == 'Logoff').sum()),
    after_hours_logon= ('is_after_hours', 'sum'),
    unique_pcs       = ('pc', 'nunique')
).reset_index()

print(f"Logon features shape:  {logon_features.shape}")

# ── Device features ───────────────────────────────────────────────────────────
device_features = device.groupby('user').agg(
    device_count   = ('activity', 'count'),
    unique_devices = ('pc', 'nunique')
).reset_index()

print(f"Device features shape: {device_features.shape}")

# ── Email features ────────────────────────────────────────────────────────────
email_features = email.groupby('user').agg(
    email_sent        = ('id', 'count'),
    avg_email_size    = ('size', 'mean'),
    total_attachments = ('attachments', 'sum'),
    unique_recipients = ('to', 'nunique')
).reset_index()

print(f"Email features shape:  {email_features.shape}")

# ── Psychometric features ─────────────────────────────────────────────────────
psych_features = psychometric[['user_id', 'O', 'C', 'E', 'A', 'N']].copy()
psych_features.columns = ['user', 'openness', 'conscientiousness',
                           'extraversion', 'agreeableness', 'neuroticism']

print(f"Psych features shape:  {psych_features.shape}")

print("\nFeature preview — logon_features:")
print(logon_features.head())

print("\nFeature preview — psych_features:")
print(psych_features.head())

print("\n" + "=" * 65)
print("STATUS: Feature engineering complete. Proceed to Step 4.")
print("=" * 65)
