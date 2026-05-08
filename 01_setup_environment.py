# =============================================================================
# 01_setup_environment.py
# Insider Threat Detection — CERT r4.2
# Step 1: Verify environment, confirm file paths, and validate dataset access
#
# Project: ML Portfolio — Project 2
# Author: Grace Egbedion
# NIW Framing: Human-factor risk analytics for cybersecurity governance
#              aligned with NIST SP 800-37, CMMC, ISO 27001
# =============================================================================

import os
import sys

print("=" * 65)
print("INSIDER THREAT DETECTION — CERT r4.2")
print("Step 1: Environment Setup and File Path Verification")
print("=" * 65)

# ── Python version ────────────────────────────────────────────────────────────
print(f"\nPython version: {sys.version}")

# ── Dataset base paths ────────────────────────────────────────────────────────
BASE    = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/r4.2/'
ANSWERS = '/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2/answers/insiders.csv'

EXPECTED_FILES = [
    'logon.csv',
    'device.csv',
    'psychometric.csv',
    'email.csv',
]

print("\nVerifying dataset files:")
all_found = True
for fname in EXPECTED_FILES:
    fpath = BASE + fname
    exists = os.path.exists(fpath)
    status = "FOUND" if exists else "MISSING"
    print(f"  [{status}] {fpath}")
    if not exists:
        all_found = False

# ── Answers file ──────────────────────────────────────────────────────────────
exists = os.path.exists(ANSWERS)
status = "FOUND" if exists else "MISSING"
print(f"  [{status}] {ANSWERS}")
if not exists:
    all_found = False

# ── Full directory listing ────────────────────────────────────────────────────
print("\nFull dataset directory listing:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if all_found:
    print("STATUS: All required files confirmed. Proceed to Step 2.")
else:
    print("WARNING: One or more files missing. Check dataset path.")
print("=" * 65)
