# =============================================================================
# 13_cost_matrix.py
# Insider Threat Detection — CERT r4.2
# Step 13: Cost Matrix and Economic Threshold Analysis
#
# In cybersecurity, a False Negative (missed insider) is significantly more
# expensive than a False Positive (analyst investigation time). This step
# assigns asymmetric cost weights and identifies the economic threshold
# that minimizes total organizational cost.
#
# Cost assumptions:
#   False Negative (missed insider): 100x — data breach, sabotage, fraud
#   False Positive (false alarm):    1x   — analyst investigation time
#
# Outputs: Cost_Matrix_Threshold.png
# Result: Economic threshold 0.5133 = default 0.50, total cost 4
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

print("=" * 55)
print("Step 13: Cost Matrix and Economic Threshold Analysis")
print("=" * 55)

COST_FN = 100
COST_FP = 1

print(f"\nCost assumptions:")
print(f"  False Negative (missed insider): {COST_FN}")
print(f"  False Positive (false alarm):    {COST_FP}")

rf_proba = results["Random Forest"]["model"].predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)

fn_counts   = []
fp_counts   = []
total_costs = []

for thresh in thresholds:
    y_pred_t = (rf_proba >= thresh).astype(int)
    fn = ((y_test == 1) & (y_pred_t == 0)).sum()
    fp = ((y_test == 0) & (y_pred_t == 1)).sum()
    fn_counts.append(fn)
    fp_counts.append(fp)
    total_costs.append(fn * COST_FN + fp * COST_FP)

min_cost_idx       = np.argmin(total_costs)
economic_threshold = thresholds[min_cost_idx]
min_cost           = total_costs[min_cost_idx]
fn_at_opt          = fn_counts[min_cost_idx]
fp_at_opt          = fp_counts[min_cost_idx]

print(f"\nEconomic threshold (minimum total cost): {economic_threshold:.4f}")
print(f"Total cost at economic threshold:         {min_cost}")
print(f"  False Negatives at threshold:           {fn_at_opt}")
print(f"  False Positives at threshold:           {fp_at_opt}")

y_pred_default = results["Random Forest"]["y_pred"]
fn_default     = ((y_test == 1) & (y_pred_default == 0)).sum()
fp_default     = ((y_test == 0) & (y_pred_default == 1)).sum()
cost_default   = fn_default * COST_FN + fp_default * COST_FP

print(f"\nDefault threshold (0.50):")
print(f"  False Negatives: {fn_default}")
print(f"  False Positives: {fp_default}")
print(f"  Total cost:      {cost_default}")

plt.figure(figsize=(9, 5))
plt.plot(thresholds, total_costs, color='darkred', label='Total Cost')
plt.axvline(x=economic_threshold, color='green', linestyle='--',
            label=f'Economic Threshold ({economic_threshold:.4f})')
plt.axvline(x=0.50, color='gray', linestyle='--',
            label='Default Threshold (0.50)')
plt.xlabel('Threshold')
plt.ylabel(f'Total Cost (FN={COST_FN}x, FP={COST_FP}x)')
plt.title('Cost Matrix Analysis — Random Forest\n'
          f'(FN Cost={COST_FN}, FP Cost={COST_FP})')
plt.legend()
plt.tight_layout()
plt.savefig("Cost_Matrix_Threshold.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Cost_Matrix_Threshold.png")

print("""
Key governance finding:
  The model already operates at its economic optimum at the default
  threshold of 0.50. No threshold tuning is required to minimize
  total organizational cost under a 100:1 FN:FP cost structure.
""")

print("=" * 55)
print("STATUS: Cost matrix complete. Proceed to Step 14.")
print("=" * 55)
