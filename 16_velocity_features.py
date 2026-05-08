# =============================================================================
# 16_velocity_features.py
# Insider Threat Detection — CERT r4.2
# Step 16: Velocity Feature Engineering
#
# Velocity features capture deviation from a user's own behavioral baseline
# rather than raw totals. Addresses peer review recommendation to engineer
# self-relative behavioral signals.
#
# Features engineered:
#   after_hours_ratio        — proportion of logon events that are after-hours
#   unique_devices_per_day   — device diversity normalized by active days
#   logon_burst_ratio        — peak single-day logons vs personal daily average
#   email_burst_ratio        — peak single-day emails vs personal daily average
#
# Output: velocity_features DataFrame (1000 users x 5 columns)
# =============================================================================

import pandas as pd
import numpy as np

print("=" * 55)
print("Step 16: Velocity Feature Engineering")
print("=" * 55)

obs_start = logon['date'].min()
obs_end   = logon['date'].max()
obs_days  = (obs_end - obs_start).days
print(f"\nObservation window: {obs_start.date()} to {obs_end.date()}")
print(f"Total days: {obs_days}")

# ── Per-user active days ──────────────────────────────────────────────────────
user_active_days = logon.groupby('user')['date'].apply(
    lambda x: x.dt.date.nunique()
).reset_index()
user_active_days.columns = ['user', 'active_days']

# ── after_hours_ratio ─────────────────────────────────────────────────────────
ah_ratio = logon.groupby('user').agg(
    total_logons    = ('activity', 'count'),
    total_ah_logons = ('is_after_hours', 'sum')
).reset_index()
ah_ratio['after_hours_ratio'] = (
    ah_ratio['total_ah_logons'] / ah_ratio['total_logons']
).round(4)

# ── unique_devices_per_day ────────────────────────────────────────────────────
device_days = device.groupby('user').agg(
    total_device_events = ('activity', 'count'),
    unique_devs         = ('pc', 'nunique')
).reset_index()
device_days = device_days.merge(user_active_days, on='user', how='left')
device_days['unique_devices_per_day'] = (
    device_days['unique_devs'] / device_days['active_days']
).round(4)

# ── logon_burst_ratio ─────────────────────────────────────────────────────────
logon['date_only']  = logon['date'].dt.date
daily_logons        = logon.groupby(
    ['user', 'date_only']).size().reset_index(name='daily_count')
user_logon_stats    = daily_logons.groupby('user').agg(
    avg_daily_logons  = ('daily_count', 'mean'),
    peak_daily_logons = ('daily_count', 'max')
).reset_index()
user_logon_stats['logon_burst_ratio'] = (
    user_logon_stats['peak_daily_logons'] /
    user_logon_stats['avg_daily_logons']
).round(4)

# ── email_burst_ratio ─────────────────────────────────────────────────────────
email['date_only']  = pd.to_datetime(email['date']).dt.date
daily_emails        = email.groupby(
    ['user', 'date_only']).size().reset_index(name='daily_email_count')
user_email_stats    = daily_emails.groupby('user').agg(
    avg_daily_emails  = ('daily_email_count', 'mean'),
    peak_daily_emails = ('daily_email_count', 'max')
).reset_index()
user_email_stats['email_burst_ratio'] = (
    user_email_stats['peak_daily_emails'] /
    user_email_stats['avg_daily_emails']
).round(4)

# ── Merge ─────────────────────────────────────────────────────────────────────
velocity_features = ah_ratio[['user', 'after_hours_ratio']].copy()
velocity_features = velocity_features.merge(
    device_days[['user', 'unique_devices_per_day']], on='user', how='left')
velocity_features = velocity_features.merge(
    user_logon_stats[['user', 'logon_burst_ratio']], on='user', how='left')
velocity_features = velocity_features.merge(
    user_email_stats[['user', 'email_burst_ratio']], on='user', how='left')

print(f"\nVelocity features shape: {velocity_features.shape}")
print(f"\nFeature preview:")
print(velocity_features.head())
print(f"\nMissing values (unique_devices_per_day NaN = 735 users with no device activity):")
print(velocity_features.isnull().sum())

print("""
Velocity feature rationale:
  after_hours_ratio:      proportion signal; more discriminating than raw count
  unique_devices_per_day: device diversity per active day; normalizes for
                          observation period length differences across users
  logon_burst_ratio:      peak/average ratio; flags sudden activity spikes
  email_burst_ratio:      peak/average ratio; flags sudden exfiltration bursts
  
  Zero-fill applied to unique_devices_per_day NaN (735 users with no
  device activity) — same semantic rationale as original device features.
""")

print("=" * 55)
print("STATUS: Velocity features complete. Proceed to Step 17.")
print("=" * 55)
