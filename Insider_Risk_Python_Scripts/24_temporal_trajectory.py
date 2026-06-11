# =============================================================================
# 24_temporal_trajectory.py
# Insider Risk Scoring — CERT r4.2
# Step 24: Temporal risk trajectory — monthly rolling risk scores
#
# Computes cumulative feature snapshots across all 17 observation months
# and scores tracked users to produce a longitudinal risk score sequence.
# Demonstrates the low-and-slow behavioral ramp-up pattern.
#
# Note: This step takes approximately 2-3 minutes to run.
#
# Key findings:
#   All 5 tracked malicious users cross Critical threshold between
#   mid-2010 and early 2011 via progressive score escalation.
#   Benign users remain near zero; one shows a temporary spike that
#   self-corrects — the false positive behavioral signature.
#
# Outputs: Risk_Trajectory_Malicious.png
#          Risk_Trajectory_Benign.png
#
# Author: Grace Egbedion
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Step 24: Temporal Risk Trajectory")
print("=" * 55)
print("(This step takes approximately 2-3 minutes)")

# ── Setup ─────────────────────────────────────────────────────────────────────
logon['year_month'] = logon['date'].dt.to_period('M')
email['date_dt']    = pd.to_datetime(email['date'])
email['year_month'] = email['date_dt'].dt.to_period('M')

months          = sorted(logon['year_month'].unique())
malicious_users = merged_v2[merged_v2['label']==1]['user'].values[:5]
benign_users    = merged_v2[merged_v2['label']==0]['user'].values[:3]
tracked_users   = list(malicious_users) + list(benign_users)

print(f"\nObservation: {months[0]} to {months[-1]} ({len(months)} months)")
print(f"Tracking: {len(malicious_users)} malicious + "
      f"{len(benign_users)} benign users")

# ── Compute monthly risk scores ───────────────────────────────────────────────
monthly_records = []

for month in months:
    logon_m  = logon[logon['year_month'] <= month]
    email_m  = email[email['year_month'] <= month]

    lf = logon_m.groupby('user').agg(
        logon_count      =('activity', lambda x: (x=='Logon').sum()),
        logoff_count     =('activity', lambda x: (x=='Logoff').sum()),
        after_hours_logon=('is_after_hours','sum'),
        unique_pcs       =('pc','nunique')
    ).reset_index()

    df_m = device.groupby('user').agg(
        device_count  =('activity','count'),
        unique_devices=('pc','nunique')
    ).reset_index()

    ef = email_m.groupby('user').agg(
        email_sent       =('id','count'),
        avg_email_size   =('size','mean'),
        total_attachments=('attachments','sum'),
        unique_recipients=('to','nunique')
    ).reset_index()

    active = logon_m.groupby('user')['date'].apply(
        lambda x: x.dt.date.nunique()).reset_index()
    active.columns = ['user','active_days']

    ahr = lf.copy()
    ahr['after_hours_ratio'] = (
        ahr['after_hours_logon'] / ahr['logon_count'].clip(lower=1))

    udpd = df_m.merge(active, on='user', how='left')
    udpd['unique_devices_per_day'] = (
        udpd['unique_devices'] / udpd['active_days'].clip(lower=1))

    lbr_d           = logon_m.copy()
    lbr_d['d_only'] = lbr_d['date'].dt.date
    dl              = lbr_d.groupby(
        ['user','d_only']).size().reset_index(name='dc')
    lbr = dl.groupby('user').agg(
        avg_dl=('dc','mean'), peak_dl=('dc','max')).reset_index()
    lbr['logon_burst_ratio'] = lbr['peak_dl']/lbr['avg_dl'].clip(lower=1)

    ebr_d           = email_m.copy()
    ebr_d['d_only'] = ebr_d['date_dt'].dt.date
    de              = ebr_d.groupby(
        ['user','d_only']).size().reset_index(name='ec')
    ebr = de.groupby('user').agg(
        avg_de=('ec','mean'), peak_de=('ec','max')).reset_index()
    ebr['email_burst_ratio'] = ebr['peak_de']/ebr['avg_de'].clip(lower=1)

    snap = psych_features.copy()
    for tbl in [lf, df_m, ef,
                ahr[['user','after_hours_ratio']],
                udpd[['user','unique_devices_per_day']],
                lbr[['user','logon_burst_ratio']],
                ebr[['user','email_burst_ratio']]]:
        snap = snap.merge(tbl, on='user', how='left')

    snap[feature_cols_v2] = snap[feature_cols_v2].fillna(0)
    snap_t  = snap[snap['user'].isin(tracked_users)]
    if len(snap_t) == 0:
        continue

    X_snap = scaler_v2.transform(snap_t[feature_cols_v2].values)
    scores = rf_v2.predict_proba(X_snap)[:, 1]

    for user, score in zip(snap_t['user'].values, scores):
        label = merged_v2[merged_v2['user']==user]['label'].values[0]
        monthly_records.append({
            'month': str(month), 'user': user,
            'risk_score': score, 'label': label
        })

traj_df      = pd.DataFrame(monthly_records)
month_labels = [str(m) for m in months]
mal_colors   = ['#C0392B','#E74C3C','#922B21','#CB4335','#A93226']
ben_colors   = ['#2E86C1','#1A5276','#117A65']

print(f"\nTrajectory records: {traj_df.shape}")

# ── Plot 1: Malicious trajectories ────────────────────────────────────────────
plt.figure(figsize=(11, 6))
for i, user in enumerate(malicious_users):
    ud = traj_df[traj_df['user']==user].sort_values('month')
    plt.plot(range(len(ud)), ud['risk_score'],
             color=mal_colors[i], linewidth=2,
             marker='o', markersize=5, label=user)
plt.axhline(y=0.75, color='red',    linestyle='--',
            linewidth=1.5, label='Critical (0.75)')
plt.axhline(y=0.50, color='orange', linestyle='--',
            linewidth=1.5, label='High (0.50)')
plt.xticks(range(len(months)), month_labels,
           rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('Risk Score', fontsize=13)
plt.xlabel('Month', fontsize=13)
plt.title('Temporal Risk Trajectory — Malicious Users', fontsize=14)
plt.legend(fontsize=11)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("Risk_Trajectory_Malicious.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Trajectory_Malicious.png")

# ── Plot 2: Benign trajectories ───────────────────────────────────────────────
plt.figure(figsize=(11, 6))
for i, user in enumerate(benign_users):
    ud = traj_df[traj_df['user']==user].sort_values('month')
    plt.plot(range(len(ud)), ud['risk_score'],
             color=ben_colors[i], linewidth=2,
             marker='o', markersize=5, label=user)
plt.axhline(y=0.75, color='red',    linestyle='--',
            linewidth=1.5, label='Critical (0.75)')
plt.axhline(y=0.50, color='orange', linestyle='--',
            linewidth=1.5, label='High (0.50)')
plt.xticks(range(len(months)), month_labels,
           rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel('Risk Score', fontsize=13)
plt.xlabel('Month', fontsize=13)
plt.title('Temporal Risk Trajectory — Benign Users', fontsize=14)
plt.legend(fontsize=11)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("Risk_Trajectory_Benign.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Risk_Trajectory_Benign.png")

print("\n" + "=" * 55)
print("STATUS: Temporal trajectory complete. Proceed to Step 25.")
print("=" * 55)
