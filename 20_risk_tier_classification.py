# =============================================================================
# 20_risk_tier_classification.py
# Insider Risk Scoring — CERT r4.2
# Step 20: Risk tier classification with lift metrics
#
# Segments all 1,000 users into four governance-aligned risk tiers and
# computes lift per tier to quantify the ROI of the risk scoring model.
#
# Tier thresholds:
#   Critical: score >= 0.75
#   High:     score >= 0.50
#   Medium:   score >= 0.25
#   Low:      score <  0.25
#
# Key results:
#   Critical tier: 98.6% malicious density, 14.1x lift
#   Low tier: 922 users, 0% malicious — 92.2% of population safely deprioritized
#
# Outputs: Risk_Tier_Classification.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 20: Risk Tier Classification with Lift")
print("=" * 55)

def assign_tier(score):
    if score >= 0.75:   return 'Critical'
    elif score >= 0.50: return 'High'
    elif score >= 0.25: return 'Medium'
    else:               return 'Low'

risk_df = pd.DataFrame({
    'user':       users_all,
    'risk_score': risk_scores_all,
    'label':      y_all,
    'tier':       [assign_tier(s) for s in risk_scores_all]
})

baseline_rate = y_all.mean()
print(f"\nPopulation baseline malicious rate: {baseline_rate:.4f} "
      f"({baseline_rate*100:.1f}%)")

tier_order = ['Critical', 'High', 'Medium', 'Low']
tier_colors_map = {'Critical':'#C0392B','High':'#E67E22',
                   'Medium':'#F1C40F','Low':'#2ECC71'}

print(f"\n{'Tier':<12} {'Users':<8} {'Malicious':<12} "
      f"{'Mal Rate':<12} {'Lift':<10} {'% of All Mal'}")
print("-" * 65)

tier_stats = []
for tier in tier_order:
    subset     = risk_df[risk_df['tier'] == tier]
    n_users    = len(subset)
    n_mal      = subset['label'].sum()
    mal_rate   = n_mal / n_users if n_users > 0 else 0
    lift       = mal_rate / baseline_rate if baseline_rate > 0 else 0
    pct_of_mal = n_mal / y_all.sum() * 100
    tier_stats.append({
        'tier': tier, 'n_users': n_users, 'n_mal': n_mal,
        'mal_rate': mal_rate, 'lift': lift, 'pct_of_mal': pct_of_mal
    })
    print(f"{tier:<12} {n_users:<8} {n_mal:<12} "
          f"{mal_rate*100:<11.1f}% {lift:<10.2f} {pct_of_mal:.1f}%")

tier_colors = [tier_colors_map[t] for t in tier_order]
counts      = [t['n_users']    for t in tier_stats]
mal_rates   = [t['mal_rate']*100 for t in tier_stats]
lifts       = [t['lift']       for t in tier_stats]

# ── Plot 1: Users per tier ────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
bars = plt.bar(tier_order, counts, color=tier_colors, edgecolor='gray')
for i, v in enumerate(counts):
    plt.text(i, v + 5, str(v), ha='center', fontsize=11)
plt.xlabel('Risk Tier', fontsize=13)
plt.ylabel('Number of Users', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.title('Users per Risk Tier', fontsize=14)
plt.tight_layout()
plt.savefig("Risk_Tier_User_Counts.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Tier_User_Counts.png")

# ── Plot 2: Malicious rate per tier ───────────────────────────────────────────
plt.figure(figsize=(9, 6))
plt.bar(tier_order, mal_rates, color=tier_colors, edgecolor='gray')
plt.axhline(y=baseline_rate*100, color='navy', linestyle='--', linewidth=1.5,
            label=f'Baseline ({baseline_rate*100:.1f}%)')
for i, v in enumerate(mal_rates):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11)
plt.xlabel('Risk Tier', fontsize=13)
plt.ylabel('Malicious Rate (%)', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.title('Malicious Rate per Risk Tier', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Tier_Malicious_Rate.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Tier_Malicious_Rate.png")

# ── Plot 3: Lift per tier ─────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
plt.bar(tier_order, lifts, color=tier_colors, edgecolor='gray')
plt.axhline(y=1.0, color='navy', linestyle='--', linewidth=1.5,
            label='Baseline lift (1.0x)')
for i, v in enumerate(lifts):
    plt.text(i, v + 0.2, f'{v:.1f}x', ha='center',
             fontsize=11, fontweight='bold')
plt.xlabel('Risk Tier', fontsize=13)
plt.ylabel('Lift', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.title('Lift per Risk Tier vs Population Baseline', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Tier_Lift.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Tier_Lift.png")

print(f"\nKey finding: Critical tier lift = {lifts[0]:.1f}x population baseline")
print("\n" + "=" * 55)
print("STATUS: Risk tier classification complete. Proceed to Step 21.")
print("=" * 55)
