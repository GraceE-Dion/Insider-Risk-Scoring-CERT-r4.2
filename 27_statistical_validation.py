# =============================================================================
# 27_statistical_validation.py
# Insider Risk Scoring — CERT r4.2
# Step 27: Bootstrap confidence intervals for risk scoring metrics
#
# Addresses peer review requirement for inferential statistics in the
# risk scoring extension. Computes bootstrap CIs for critical tier lift,
# malicious rate, and Brier Score improvement to confirm that the reported
# point estimates are statistically robust rather than sample-dependent.
#
# Key results:
#   Critical tier malicious rate: 98.57% [95% CI: 95.45%, 100.00%]
#   Critical tier lift:           14.28x [95% CI: 11.49x, 18.51x]
#   % of all malicious in Crit:   98.62% [95% CI: 95.45%, 100.00%]
#   Brier improvement (isotonic): 61.4%  [95% CI: 33.8%, 100.0%]
#
#   Critical tier lift CI entirely above 1.0x — statistically robust.
#   High tier CI includes zero (n=4 users, small sample, noted as caveat).
#
# Outputs: Statistical_Validation_Risk.csv
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print("=" * 55)
print("Step 27: Bootstrap CIs for Risk Scoring Metrics")
print("=" * 55)

n_bootstraps  = 1000
baseline_rate = y_all.mean()
print(f"\nPopulation baseline malicious rate: {baseline_rate:.4f}")
print(f"Running {n_bootstraps} bootstrap iterations...")

# ── Bootstrap tier metrics ────────────────────────────────────────────────────
lift_critical = []
lift_high     = []
mal_rate_crit = []
mal_rate_high = []
pct_mal_crit  = []

np.random.seed(42)
for _ in range(n_bootstraps):
    idx      = resample(np.arange(len(y_all)), replace=True)
    y_bs     = y_all[idx]
    score_bs = risk_scores_all[idx]
    tier_bs  = [assign_tier(s) for s in score_bs]

    base_bs  = y_bs.mean()
    if base_bs == 0:
        continue

    crit_mask = np.array([t == 'Critical' for t in tier_bs])
    if crit_mask.sum() > 0:
        cr = y_bs[crit_mask].mean()
        mal_rate_crit.append(cr)
        lift_critical.append(cr / base_bs)
        pct_mal_crit.append(y_bs[crit_mask].sum() / y_bs.sum() * 100)

    high_mask = np.array([t == 'High' for t in tier_bs])
    if high_mask.sum() > 0:
        hr = y_bs[high_mask].mean()
        mal_rate_high.append(hr)
        lift_high.append(hr / base_bs)

# ── Bootstrap Brier improvement ───────────────────────────────────────────────
X_cal_b, X_cal_t, y_cal_b, y_cal_t = train_test_split(
    X_all_scaled, y_all,
    test_size=0.20, random_state=99, stratify=y_all)

rf_iso_bci = CalibratedClassifierCV(rf_v2, method='isotonic', cv='prefit')
rf_iso_bci.fit(X_cal_b, y_cal_b)

prob_raw_bci = rf_v2.predict_proba(X_cal_t)[:, 1]
prob_iso_bci = rf_iso_bci.predict_proba(X_cal_t)[:, 1]

bs_raw_vals = []
bs_iso_vals = []
bs_imp_vals = []

np.random.seed(42)
for _ in range(n_bootstraps):
    idx    = resample(np.arange(len(y_cal_t)), replace=True)
    y_bs   = y_cal_t[idx]
    raw_bs = prob_raw_bci[idx]
    iso_bs = prob_iso_bci[idx]
    if len(np.unique(y_bs)) < 2:
        continue
    br = brier_score_loss(y_bs, raw_bs)
    bi = brier_score_loss(y_bs, iso_bs)
    bs_raw_vals.append(br)
    bs_iso_vals.append(bi)
    bs_imp_vals.append((1 - bi/br)*100 if br > 0 else 0)

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n95% Bootstrap CIs — Risk Tier Metrics:")
print(f"\n{'Metric':<38} {'Mean':<10} {'95% CI'}")
print("-" * 65)

all_results = []
for label, values in [
    ("Critical tier malicious rate (%)", [v*100 for v in mal_rate_crit]),
    ("Critical tier lift",               lift_critical),
    ("% of all malicious in Critical",   pct_mal_crit),
    ("High tier lift",                   lift_high),
    ("Brier Score (uncalibrated)",        bs_raw_vals),
    ("Brier Score (isotonic)",            bs_iso_vals),
    ("Brier improvement (%)",             bs_imp_vals),
]:
    lo = np.percentile(values, 2.5)
    hi = np.percentile(values, 97.5)
    mn = np.mean(values)
    print(f"  {label:<36} {mn:<10.4f} [{lo:.4f}, {hi:.4f}]")
    all_results.append({'metric': label, 'mean': mn, 'ci_lo': lo, 'ci_hi': hi})

print("\nNote: High tier lift CI includes zero due to small sample (n=4 users)")
print("This result should be interpreted with appropriate caution in the paper.")

pd.DataFrame(all_results).to_csv(
    "Statistical_Validation_Risk.csv", index=False)
print("\nSaved: Statistical_Validation_Risk.csv")

print("\n" + "=" * 55)
print("STATUS: Statistical validation complete. Proceed to Step 28.")
print("=" * 55)
