# =============================================================================
# 02_load_data.py
# Insider Threat Detection — CERT r4.2
# Step 2: Load core activity files and inspect structure
#
# Loads: logon.csv, device.csv, psychometric.csv, insiders.csv (answers)
# Outputs: logon, device, psychometric, insiders DataFrames
# =============================================================================

import pandas as pd

print("=" * 65)
print("Step 2: Load Core Activity Files")
print("=" * 65)

BASE    = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/r4.2/'
ANSWERS = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/answers/insiders.csv'

# ── Load files ────────────────────────────────────────────────────────────────
logon        = pd.read_csv(BASE + 'logon.csv')
device       = pd.read_csv(BASE + 'device.csv')
psychometric = pd.read_csv(BASE + 'psychometric.csv')
insiders     = pd.read_csv(ANSWERS)

# ── Shape inspection ──────────────────────────────────────────────────────────
print(f"\nLogon shape:        {logon.shape}")
print(f"Device shape:       {device.shape}")
print(f"Psychometric shape: {psychometric.shape}")
print(f"Insiders shape:     {insiders.shape}")

print(f"\nInsiders head (10):")
print(insiders.head(10))

print(f"\nLogon columns:   {logon.columns.tolist()}")
print(f"Device columns:  {device.columns.tolist()}")

# ── Load email (large file — load separately) ─────────────────────────────────
print("\nLoading email.csv (2.6M rows — may take a moment)...")
email = pd.read_csv(BASE + 'email.csv')
print(f"Email shape: {email.shape}")
print(f"Email columns: {email.columns.tolist()}")

# ── Insider scenarios breakdown ───────────────────────────────────────────────
print("\nInsider scenarios breakdown:")
print(insiders['scenario'].value_counts())

print(f"\nUnique malicious users: {insiders['user'].nunique()}")

print("\nSample logon activities:")
print(logon['activity'].value_counts())

print("\n" + "=" * 65)
print("STATUS: Data loaded successfully. Proceed to Step 3.")
print("=" * 65)
