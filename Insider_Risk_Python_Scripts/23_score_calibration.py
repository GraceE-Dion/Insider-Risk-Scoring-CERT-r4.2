# =============================================================================
# 23_score_calibration.py
# Insider Risk Scoring — CERT r4.2
# Step 23: Score calibration — Isotonic Regression and Platt Scaling
#
# Calibrates predict_proba outputs so a score of 0.7 corresponds to
# approximately 70% empirical likelihood — required for audit-readiness
# under NIST AI RMF 1.0 for systems informing personnel risk decisions.
#
# Key results:
#   Uncalibrated Brier Score: 0.0038
#   Isotonic Brier Score:     0.0022 (42% improvement)
#   Platt Scaling:            0.0037 (marginal)
#   Recommended: Isotonic calibration for production scoring
#
# Outputs: Risk_Score_Calibration_Curve.png
#          Risk_Score_Distribution_Calibrated.png
#
# Author: Grace Egbedion
# =============================================================================

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 23: Score Calibration")
print("=" * 55)

# ── Calibration split ─────────────────────────────────────────────────────────
X_cal_base, X_cal_test, y_cal_base, y_cal_test = train_test_split(
    X_all_scaled, y_all,
    test_size=0.20, random_state=99, stratify=y_all)

# ── Fit calibration models ────────────────────────────────────────────────────
rf_iso   = CalibratedClassifierCV(rf_v2, method='isotonic', cv='prefit')
rf_platt = CalibratedClassifierCV(rf_v2, method='sigmoid',  cv='prefit')
rf_iso.fit(X_cal_base,   y_cal_base)
rf_platt.fit(X_cal_base, y_cal_base)

prob_raw   = rf_v2.predict_proba(X_cal_test)[:, 1]
prob_iso   = rf_iso.predict_proba(X_cal_test)[:, 1]
prob_platt = rf_platt.predict_proba(X_cal_test)[:, 1]

frac_pos_raw,   mean_pred_raw   = calibration_curve(
    y_cal_test, prob_raw,   n_bins=10, strategy='quantile')
frac_pos_iso,   mean_pred_iso   = calibration_curve(
    y_cal_test, prob_iso,   n_bins=10, strategy='quantile')
frac_pos_platt, mean_pred_platt = calibration_curve(
    y_cal_test, prob_platt, n_bins=10, strategy='quantile')

bs_raw   = brier_score_loss(y_cal_test, prob_raw)
bs_iso   = brier_score_loss(y_cal_test, prob_iso)
bs_platt = brier_score_loss(y_cal_test, prob_platt)

print(f"\nBrier Score (lower = better):")
print(f"  Uncalibrated: {bs_raw:.4f}")
print(f"  Isotonic:     {bs_iso:.4f}  "
      f"({(1-bs_iso/bs_raw)*100:.1f}% improvement)")
print(f"  Platt:        {bs_platt:.4f}  "
      f"({(1-bs_platt/bs_raw)*100:.1f}% improvement)")

# ── Plot 1: Calibration curves ────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot([0,1],[0,1],'k--',linewidth=1.5,label='Perfect calibration')
plt.plot(mean_pred_raw,   frac_pos_raw,   'o-', color='steelblue',
         linewidth=2, markersize=6,
         label=f'Uncalibrated (Brier={bs_raw:.4f})')
plt.plot(mean_pred_iso,   frac_pos_iso,   's-', color='crimson',
         linewidth=2, markersize=6,
         label=f'Isotonic (Brier={bs_iso:.4f})')
plt.plot(mean_pred_platt, frac_pos_platt, '^-', color='darkorange',
         linewidth=2, markersize=6,
         label=f'Platt Scaling (Brier={bs_platt:.4f})')
plt.xlabel('Mean Predicted Probability', fontsize=13)
plt.ylabel('Fraction of Positives', fontsize=13)
plt.xticks(fontsize=11); plt.yticks(fontsize=11)
plt.title('Calibration Curves — Reliability Diagram', fontsize=14)
plt.legend(fontsize=11); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Risk_Score_Calibration_Curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Score_Calibration_Curve.png")

# ── Plot 2: Score distributions ───────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.hist(prob_raw[y_cal_test==0], bins=30, alpha=0.5,
         color='steelblue', label='Benign (uncalibrated)')
plt.hist(prob_iso[y_cal_test==0], bins=30, alpha=0.5,
         color='lightblue', label='Benign (isotonic)')
plt.hist(prob_raw[y_cal_test==1], bins=20, alpha=0.6,
         color='crimson', label='Malicious (uncalibrated)')
plt.hist(prob_iso[y_cal_test==1], bins=20, alpha=0.6,
         color='salmon', label='Malicious (isotonic)')
plt.xlabel('Risk Score', fontsize=13); plt.ylabel('Count', fontsize=13)
plt.xticks(fontsize=11); plt.yticks(fontsize=11)
plt.title('Score Distribution — Uncalibrated vs Isotonic', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Score_Distribution_Calibrated.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Score_Distribution_Calibrated.png")

print("\n" + "=" * 55)
print("STATUS: Calibration complete. Proceed to Step 24.")
print("=" * 55)
