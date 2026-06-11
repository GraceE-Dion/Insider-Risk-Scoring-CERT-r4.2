# =============================================================================
# 25_risk_velocity.py
# Insider Risk Scoring — CERT r4.2
# Step 25: Risk velocity — monthly score change alerting
#
# Computes month-over-month risk score change (velocity) for tracked users.
# Velocity threshold: +0.10 per month.
#
# Key results:
#   18 high-velocity events detected
#   15 (83%) belong to malicious users
#   Concentration: July to October 2010 (behavioral acceleration phase)
#   Benign velocity tightly clustered near zero
#
# Outputs: Risk_Velocity_Malicious.png
#          Risk_Velocity_Distribution.png
#
# Author: Grace Egbedion
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 25: Risk Velocity")
print("=" * 55)

# ── Compute velocity ──────────────────────────────────────────────────────────
velocity_records = []
for user in tracked_users:
    ud    = traj_df[traj_df['user']==user].sort_values(
        'month').reset_index(drop=True)
    label = ud['label'].iloc[0]
    for i in range(1, len(ud)):
        velocity_records.append({
            'user':     user,
            'month':    ud['month'].iloc[i],
            'score':    ud['risk_score'].iloc[i],
            'velocity': ud['risk_score'].iloc[i] - ud['risk_score'].iloc[i-1],
            'label':    label
        })

vel_df        = pd.DataFrame(velocity_records)
VEL_THRESHOLD = 0.10
high_vel      = vel_df[vel_df['velocity'] >= VEL_THRESHOLD]

print(f"\nVelocity alert threshold: +{VEL_THRESHOLD} per month")
print(f"High velocity events detected: {len(high_vel)}")
print(f"  Malicious: {high_vel['label'].sum()}")
print(f"  Benign:    {(high_vel['label']==0).sum()}")
print(f"\nHigh velocity events detail:")
print(high_vel[['user','month','score','velocity','label']].to_string(
    index=False))

mal_colors = ['#C0392B','#E74C3C','#922B21','#CB4335','#A93226']

# ── Plot 1: Velocity over time — malicious users ──────────────────────────────
plt.figure(figsize=(11, 6))
for i, user in enumerate(malicious_users):
    ud = vel_df[vel_df['user']==user]
    plt.plot(range(len(ud)), ud['velocity'],
             color=mal_colors[i], linewidth=2,
             marker='o', markersize=5, label=user)
plt.axhline(y=VEL_THRESHOLD, color='red', linestyle='--',
            linewidth=1.5, label=f'Alert threshold (+{VEL_THRESHOLD})')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
plt.xticks(range(len(months)-1),
           [str(m) for m in months[1:]],
           rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('Risk Score Change (Monthly)', fontsize=13)
plt.xlabel('Month', fontsize=13)
plt.title('Risk Velocity — Malicious Users\n(Monthly score change)',
          fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Velocity_Malicious.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Velocity_Malicious.png")

# ── Plot 2: Velocity distribution ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
mal_vel = vel_df[vel_df['label']==1]['velocity']
ben_vel = vel_df[vel_df['label']==0]['velocity']
plt.hist(ben_vel, bins=20, alpha=0.6, color='steelblue',
         label='Benign', density=True)
plt.hist(mal_vel, bins=20, alpha=0.6, color='crimson',
         label='Malicious', density=True)
plt.axvline(x=VEL_THRESHOLD, color='red', linestyle='--',
            linewidth=1.5, label=f'Alert threshold (+{VEL_THRESHOLD})')
plt.xlabel('Monthly Risk Score Change', fontsize=13)
plt.ylabel('Density', fontsize=13)
plt.xticks(fontsize=11); plt.yticks(fontsize=11)
plt.title('Risk Velocity Distribution — Malicious vs Benign', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("Risk_Velocity_Distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Velocity_Distribution.png")

print("\n" + "=" * 55)
print("STATUS: Risk velocity complete. Proceed to Step 26.")
print("=" * 55)
