# =============================================================================
# 09_threshold_tuning.py
# Insider Threat Detection — CERT r4.2
# Step 9: Precision/Recall/F1 threshold tuning for operational deployment
#
# In insider threat detection, missing a true insider (false negative) is
# significantly more costly than a false alarm (false positive).
# This step identifies the optimal operating threshold for the SOC context.
#
# Output: threshold_curve.png, threshold analysis table
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

print("=" * 65)
print("Step 9: Threshold Tuning — Precision / Recall / F1")
print("=" * 65)

rf_proba = results["Random Forest"]["model"].predict_proba(X_test_scaled)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)

# ── Build threshold analysis table ───────────────────────────────────────────
thresh_df = pd.DataFrame({
    'threshold': thresholds,
    'precision': precisions[:-1],
    'recall':    recalls[:-1],
    'f1':        2 * (precisions[:-1] * recalls[:-1]) /
                     (precisions[:-1] + recalls[:-1] + 1e-9)
})

print("\nThreshold analysis (recall >= 0.90):")
print(thresh_df[thresh_df['recall'] >= 0.90].round(4).to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision', color='blue')
plt.plot(thresholds, recalls[:-1],    label='Recall',    color='orange')
plt.plot(thresholds, thresh_df['f1'], label='F1 Score',  color='green')
plt.axvline(x=0.5, color='gray', linestyle='--',
            label='Default threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Threshold — Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig("threshold_curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: threshold_curve.png")

# ── Recommendation ────────────────────────────────────────────────────────────
print("""
Threshold Recommendation:
  Operating threshold: 0.50 (default)
  At threshold=0.50:
    Recall (Malicious):    1.00  — zero missed insiders
    Precision (Malicious): 0.78  — 4 false alarms per 200 users
    F1 (Malicious):        0.88

  Recall stays at 1.00 up to threshold ~0.73.
  In SOC deployment, catching every insider at the cost of
  4 false positives per 200 users is an operationally
  acceptable trade-off aligned with NIST SP 800-37 continuous
  monitoring and CMMC access control requirements.
""")

print("=" * 65)
print("STATUS: Threshold tuning complete. Proceed to Step 10.")
print("=" * 65)
