# =============================================================================
# 19_risk_score_distribution.py
# Insider Risk Scoring — CERT r4.2
# Step 19: Risk score distribution for all 1,000 users
#
# Computes continuous risk scores for all 1,000 users using the pre-fitted
# scaler and enhanced Random Forest model. Produces histogram and KDE
# density plots showing near-perfect separation between populations.
#
# Key results:
#   Benign mean score:    0.0077
#   Malicious mean score: 0.9715
#   Score range:          0.0000 to 1.0000
#
# Outputs: Risk_Score_Distribution.png
#
# Author: Grace Egbedion
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

print("=" * 55)
print("Step 19: Risk Score Distribution")
print("=" * 55)

# ── Compute risk scores for all 1,000 users ───────────────────────────────────
X_all         = merged_v2[feature_cols_v2].values
y_all         = merged_v2['label'].values
users_all     = merged_v2['user'].values
X_all_scaled  = scaler_v2.transform(X_all)

risk_scores_all = rf_v2.predict_proba(X_all_scaled)[:, 1]

print(f"\nRisk scores computed for {len(risk_scores_all)} users")
print(f"Score range: {risk_scores_all.min():.4f} to "
      f"{risk_scores_all.max():.4f}")
print(f"Mean score (benign):    "
      f"{risk_scores_all[y_all==0].mean():.4f}")
print(f"Mean score (malicious): "
      f"{risk_scores_all[y_all==1].mean():.4f}")

# ── Plot 1: Histogram ─────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.hist(risk_scores_all[y_all==0], bins=40, alpha=0.6,
         color='steelblue', label='Benign (930 users)')
plt.hist(risk_scores_all[y_all==1], bins=40, alpha=0.7,
         color='crimson', label='Malicious (70 users)')
for thresh, color, label in [
        (0.25, 'orange',     'Medium threshold (0.25)'),
        (0.50, 'darkorange', 'High threshold (0.50)'),
        (0.75, 'red',        'Critical threshold (0.75)')]:
    plt.axvline(x=thresh, color=color, linestyle='--',
                linewidth=1.5, label=label)
plt.xlabel('Risk Score', fontsize=13)
plt.ylabel('Number of Users', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Risk Score Distribution — All 1,000 Users', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Score_Distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Score_Distribution.png")

# ── Plot 2: KDE density ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for label, color, name in [(0,'steelblue','Benign'),
                             (1,'crimson','Malicious')]:
    scores  = risk_scores_all[y_all==label]
    kde     = gaussian_kde(scores, bw_method=0.15)
    x_range = np.linspace(0, 1, 300)
    plt.plot(x_range, kde(x_range), color=color, linewidth=2, label=name)
    plt.fill_between(x_range, kde(x_range), alpha=0.15, color=color)

for thresh, color in [(0.25,'orange'),(0.50,'darkorange'),(0.75,'red')]:
    plt.axvline(x=thresh, color=color, linestyle='--', linewidth=1.5)
plt.xlabel('Risk Score', fontsize=13)
plt.ylabel('Density', fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Risk Score Density — Benign vs Malicious', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Score_Density.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Score_Density.png")

print("\n" + "=" * 55)
print("STATUS: Risk score distribution complete. Proceed to Step 20.")
print("=" * 55)
