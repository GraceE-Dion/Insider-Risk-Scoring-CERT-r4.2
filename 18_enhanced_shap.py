# =============================================================================
# 18_enhanced_shap.py
# Insider Threat Detection — CERT r4.2
# Step 18: SHAP Explainability — Enhanced Model (19 Features)
#
# Recomputes SHAP values for the enhanced 19-feature Random Forest model.
# Key finding: unique_devices_per_day (velocity feature) rises to rank 1
# with SHAP impact 0.1895, displacing the original unique_devices.
# Normalized device diversity is more discriminating than raw device count.
#
# All four velocity features rank in the top 8.
# Psychometric features remain negligible (all below 0.003).
#
# Outputs: SHAP_Summary_Enhanced_Model.png
#          SHAP_Bar_Enhanced_Model.png
# =============================================================================

import shap
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 18: SHAP Explainability — Enhanced Model (19 Features)")
print("=" * 55)

explainer_v2     = shap.TreeExplainer(rf_v2)
shap_values_v2   = explainer_v2.shap_values(X_test_v2_scaled)
shap_vals_v2_mal = shap_values_v2[:, :, 1]

print(f"SHAP values shape: {np.array(shap_values_v2).shape}")

# ── Beeswarm summary plot ─────────────────────────────────────────────────────
plt.figure()
shap.summary_plot(
    shap_vals_v2_mal,
    X_test_v2_scaled,
    feature_names=feature_cols_v2,
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary — Enhanced Model (19 Features)")
plt.tight_layout()
plt.savefig("SHAP_Summary_Enhanced_Model.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: SHAP_Summary_Enhanced_Model.png")

# ── Bar plot ──────────────────────────────────────────────────────────────────
plt.figure()
shap.summary_plot(
    shap_vals_v2_mal,
    X_test_v2_scaled,
    feature_names=feature_cols_v2,
    plot_type="bar",
    show=False
)
plt.title("SHAP Mean Absolute Impact — Enhanced Model (19 Features)")
plt.tight_layout()
plt.savefig("SHAP_Bar_Enhanced_Model.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: SHAP_Bar_Enhanced_Model.png")

# ── SHAP ranking ──────────────────────────────────────────────────────────────
velocity_cols = ['after_hours_ratio', 'unique_devices_per_day',
                 'logon_burst_ratio', 'email_burst_ratio']
mean_shap     = np.abs(shap_vals_v2_mal).mean(axis=0)
shap_df       = sorted(zip(feature_cols_v2, mean_shap),
                        key=lambda x: x[1], reverse=True)

print("\nSHAP mean absolute impact — all 19 features:")
for rank, (feat, val) in enumerate(shap_df, 1):
    marker = " <-- NEW" if feat in velocity_cols else ""
    print(f"  {rank:>2}. {feat:<35} {val:.4f}{marker}")

print("""
Key SHAP findings — enhanced model:
  unique_devices_per_day rises to rank 1 (0.1895) — normalized device
  diversity is more discriminating than the raw count (now rank 4, 0.0656).
  All 4 velocity features rank in the top 8.
  Psychometric features remain below 0.003 — conclusion unchanged.
""")

print("=" * 55)
print("STATUS: Enhanced SHAP complete. All Steps 12-18 done.")
print("=" * 55)
