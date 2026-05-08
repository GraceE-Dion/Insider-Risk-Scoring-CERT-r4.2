# =============================================================================
# 22_risk_user_report.py
# Insider Risk Scoring — CERT r4.2
# Step 22: Top-N risk user report with SHAP-driven explanations
#
# Generates a ranked risk report for all 1,000 users showing risk score,
# tier, actual label, and top 3 behavioral drivers per user. This simulates
# the SOC dashboard output — an analyst sees not just a score but the
# behavioral story behind it.
#
# Output: Insider_Risk_Report.csv (1,000 users, ranked by risk score)
# =============================================================================

import pandas as pd
import numpy as np

print("=" * 55)
print("Step 22: Top-N Risk User Report")
print("=" * 55)

print("\nComputing SHAP for all 1,000 users (may take ~30 seconds)...")
shap_all     = explainer_v2.shap_values(X_all_scaled)
shap_all_mal = shap_all[:, :, 1]

def top3_drivers(shap_row, feature_names):
    indices = np.argsort(np.abs(shap_row))[::-1][:3]
    return ", ".join([feature_names[i] for i in indices])

top3 = [top3_drivers(shap_all_mal[i], feature_cols_v2)
        for i in range(len(shap_all_mal))]

report_df = pd.DataFrame({
    'user':          users_all,
    'risk_score':    risk_scores_all.round(4),
    'risk_tier':     [assign_tier(s) for s in risk_scores_all],
    'actual_label':  ['MALICIOUS' if l == 1 else 'Benign' for l in y_all],
    'top_3_drivers': top3
})

report_df = report_df.sort_values(
    'risk_score', ascending=False).reset_index(drop=True)

print(f"\nTop 20 Highest Risk Users:")
print(f"{'Rank':<6} {'User':<12} {'Score':<8} {'Tier':<10} "
      f"{'Actual':<12} {'Top 3 Drivers'}")
print("-" * 85)
for i, row in report_df.head(20).iterrows():
    print(f"{i+1:<6} {row['user']:<12} {row['risk_score']:<8} "
          f"{row['risk_tier']:<10} {row['actual_label']:<12} "
          f"{row['top_3_drivers']}")

print(f"\nTier distribution:")
print(report_df['risk_tier'].value_counts())

report_df.to_csv("Insider_Risk_Report.csv", index=False)
print("\nSaved: Insider_Risk_Report.csv")

print("\n" + "=" * 55)
print("STATUS: Risk user report complete. Proceed to Step 23.")
print("=" * 55)
