# =============================================================================
# 21_shap_tier_analysis.py
# Insider Risk Scoring — CERT r4.2
# Step 21: Per-tier SHAP driver analysis and explainability gap
#
# Computes mean absolute SHAP impact per risk tier to identify which
# behavioral features drive users into each tier. Also computes the
# Explainability Gap: the difference in SHAP profiles between True Positives
# (actual insiders correctly flagged) and False Positives (benign users
# incorrectly flagged) in the Critical/High tiers.
#
# Key findings:
#   Critical tier dominated by unique_devices_per_day and logoff_count
#   device_count is the primary differentiator: TP 0.0645 vs FP 0.0248
#   False positives driven by logoff_count and after_hours_ratio
#
# Outputs: SHAP_Tier_Analysis.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 21: Per-Tier SHAP Driver Analysis")
print("=" * 55)

# ── Test set tiers ────────────────────────────────────────────────────────────
test_scores = rf_v2.predict_proba(X_test_v2_scaled)[:, 1]
test_tiers  = [assign_tier(s) for s in test_scores]

top_features = [feature_cols_v2[i] for i in
                np.argsort(np.abs(shap_vals_v2_mal).mean(axis=0))[::-1][:8]]

# ── Mean SHAP per tier ────────────────────────────────────────────────────────
tier_shap = {}
for tier in ['Critical', 'High', 'Medium', 'Low']:
    idx = [i for i, t in enumerate(test_tiers) if t == tier]
    if len(idx) > 0:
        tier_shap[tier] = np.abs(shap_vals_v2_mal[idx]).mean(axis=0)
    else:
        tier_shap[tier] = np.zeros(len(feature_cols_v2))

print("\nMean absolute SHAP impact by tier (top 8 features):")
header = f"{'Feature':<30}" + "".join(
    f"{t:<12}" for t in ['Critical','High','Medium','Low'])
print(header)
print("-" * 78)
for feat in top_features:
    idx = feature_cols_v2.index(feat)
    row = f"{feat:<30}"
    for tier in ['Critical','High','Medium','Low']:
        row += f"{tier_shap[tier][idx]:<12.4f}"
    print(row)

# ── Explainability gap ────────────────────────────────────────────────────────
print("\n--- Explainability Gap: True Positives vs False Positives ---")
tp_idx = [i for i, (t, l) in enumerate(zip(test_tiers, y_test_v2))
          if t in ['Critical','High'] and l == 1]
fp_idx = [i for i, (t, l) in enumerate(zip(test_tiers, y_test_v2))
          if t in ['Critical','High'] and l == 0]

print(f"True Positives in Critical/High:  {len(tp_idx)}")
print(f"False Positives in Critical/High: {len(fp_idx)}")

tp_shap = np.abs(shap_vals_v2_mal[tp_idx]).mean(axis=0)
fp_shap = np.abs(shap_vals_v2_mal[fp_idx]).mean(axis=0)

print(f"\n{'Feature':<30} {'True Pos':<12} {'False Pos':<12} {'Difference'}")
print("-" * 65)
for feat in top_features:
    idx  = feature_cols_v2.index(feat)
    diff = tp_shap[idx] - fp_shap[idx]
    print(f"{feat:<30} {tp_shap[idx]:<12.4f} {fp_shap[idx]:<12.4f} {diff:+.4f}")

# ── Plot 1: SHAP drivers by tier ──────────────────────────────────────────────
x = np.arange(len(top_features))
w = 0.2

plt.figure(figsize=(12, 6))
for i, tier in enumerate(['Critical','High','Medium','Low']):
    vals = [tier_shap[tier][feature_cols_v2.index(f)] for f in top_features]
    plt.bar(x + i*w, vals, w, label=tier,
            color=tier_colors_map[tier], alpha=0.85)

plt.xticks(x + w*1.5, top_features, rotation=35, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('Mean Absolute SHAP Impact', fontsize=13)
plt.title('SHAP Feature Drivers by Risk Tier', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("SHAP_Tier_Drivers.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: SHAP_Tier_Drivers.png")

# ── Plot 2: Explainability gap ────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
tp_vals = [tp_shap[feature_cols_v2.index(f)] for f in top_features]
fp_vals = [fp_shap[feature_cols_v2.index(f)] for f in top_features]
plt.bar(x - 0.2, tp_vals, 0.4, label='True Positive',
        color='crimson', alpha=0.8)
plt.bar(x + 0.2, fp_vals, 0.4, label='False Positive',
        color='steelblue', alpha=0.8)
plt.xticks(x, top_features, rotation=35, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('Mean Absolute SHAP Impact', fontsize=13)
plt.title('Explainability Gap: True Positive vs False Positive Drivers',
          fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("SHAP_Explainability_Gap.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: SHAP_Explainability_Gap.png")

print("\n" + "=" * 55)
print("STATUS: SHAP tier analysis complete. Proceed to Step 22.")
print("=" * 55)
