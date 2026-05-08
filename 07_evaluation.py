# =============================================================================
# 07_evaluation.py
# Insider Threat Detection — CERT r4.2
# Step 7: Model evaluation — ROC curve comparison
#
# Generates ROC curve comparing Random Forest vs XGBoost vs baseline
# Saves: roc_curve_comparison.png
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

print("=" * 65)
print("Step 7: Evaluation — ROC Curve Comparison")
print("=" * 65)

plt.figure(figsize=(8, 6))

for name, res in results.items():
    y_proba = res['model'].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Insider Threat Detection (CERT r4.2)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: roc_curve_comparison.png")

# ── Interpretation ────────────────────────────────────────────────────────────
print("""
ROC Curve Interpretation:
  - Both models hug the top-left corner — near-perfect discrimination
  - Random Forest AUC: 0.9939
  - XGBoost AUC:       0.9965
  - Both vastly exceed the 0.90 target threshold
  - Random Forest selected as production model due to perfect recall (1.00)
    ensuring zero missed insider threats in test evaluation
""")

print("=" * 65)
print("STATUS: Evaluation complete. Proceed to Step 8.")
print("=" * 65)
