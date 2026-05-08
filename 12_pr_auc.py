# =============================================================================
# 12_pr_auc.py
# Insider Threat Detection — CERT r4.2
# Step 12: Precision-Recall AUC Comparison
#
# PR-AUC is a stricter and more honest metric than ROC-AUC for highly
# imbalanced data (7% malicious class). Addresses auditor skepticism
# about perfect recall results on imbalanced benchmarks.
#
# Outputs: PR_Curve_Comparison.png
# Results: RF PR-AUC 0.9167 | XGBoost PR-AUC 0.9618 | Baseline 0.0700
# =============================================================================

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 12: Precision-Recall AUC Comparison")
print("=" * 55)

plt.figure(figsize=(8, 6))

for name, res in results.items():
    y_proba = res['model'].predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.plot(recall, precision, label=f"{name} (PR-AUC={ap:.4f})")
    print(f"{name} PR-AUC: {ap:.4f}")

# Baseline — random classifier reflects true class imbalance
baseline = y_test.sum() / len(y_test)
plt.axhline(y=baseline, color='k', linestyle='--',
            label=f'Random Baseline (PR-AUC={baseline:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Insider Threat Detection (CERT r4.2)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("PR_Curve_Comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: PR_Curve_Comparison.png")

print("""
Interpretation:
  RF PR-AUC 0.9167 vs XGBoost 0.9618 vs Random Baseline 0.0700
  Both models vastly outperform the 0.07 random baseline.
  The 13x improvement over baseline is the true measure of model
  value on imbalanced data — far more informative than ROC-AUC
  for a 7% minority class.
""")

print("=" * 55)
print("STATUS: PR-AUC complete. Proceed to Step 13.")
print("=" * 55)
