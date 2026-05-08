# Insider Risk Scoring — CERT r4.2

**Continuous behavioral risk quantification for cybersecurity governance**
Aligned with NIST SP 800-37 | CMMC | ISO 27001 | NIST AI RMF 1.0

> ML Portfolio Project 2 (Extension) | Author: Grace Egbedion | PhD Candidate, Computational and Data Science, MTSU

---

## Project Summary

This project extends the [Insider Threat Detection](https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2) binary classification pipeline into a **continuous risk scoring system** that assigns every user a risk score between 0 and 1, segments them into governance-aligned risk tiers, and explains what behavioral signals are driving each user's risk level.

The shift from binary detection to continuous scoring is not merely a technical extension. It represents a fundamental alignment with the **risk-based approach** mandated by ISO 27001 and NIST SP 800-37: instead of answering "is this user malicious?" the system answers "how risky is this user, why, and how is that risk changing over time?" This framing enables proportional governance responses — monitoring, restricted access, coaching, or escalation — rather than a binary flagged/not-flagged output that offers no operational nuance.

---

## Relationship to Insider Threat Detection Project

This project builds directly on:
[GraceE-Dion/Insider-Threat-Detection-CERT-r4.2](https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2)

Scripts 01-18 are inherited from that project. Scripts 19-26 implement the continuous risk scoring extension. Together the two projects represent a complete human-factor behavioral analytics pipeline spanning binary detection and continuous risk quantification.

---

## Dataset

### CERT Insider Threat Dataset r4.2

Same dataset as the Insider Threat Detection project: Carnegie Mellon University SEI synthetic behavioral dataset, 1,000 simulated users, 70 malicious (7%), 17-month observation window.

See the Insider Threat Detection project for full dataset documentation.

Kaggle dataset: https://www.kaggle.com/datasets/andrihjonior/cert-insider-threat-dataset-r4-2

---

## Development Session Narrative

### Session 1: Binary Classification Pipeline (Inherited)

The full feature engineering, preprocessing, model training, SHAP explainability, PR-AUC, cost matrix, adversarial sensitivity testing, and velocity feature enhancement pipeline from the Insider Threat Detection project is re-used here without modification. The enhanced Random Forest model (rf_v2) trained on 19 behavioral and psychometric features with dual class balancing (SMOTE combined with class_weight='balanced') produces the predict_proba outputs that serve as the foundation for all risk scoring work.

Key inherited results: AUC 0.9985, PR-AUC 0.9841, Recall 1.00, CV AUC 0.9991 +/- 0.0013.

### Session 2: Continuous Risk Scoring Extension

**Objective:** Transform binary classification outputs into an operationally actionable continuous risk scoring system with tier segmentation, calibration, temporal trajectory modeling, velocity alerting, and feature stability documentation.

#### From Binary to Spectrum

The binary classifier's predict_proba output already produces a continuous value between 0 and 1 for every user. The risk scoring extension makes this continuous output the primary deliverable rather than the binary threshold crossing. This is not a trivial reframing. A user with a score of 0.4 who has been rising at +0.15 per month is operationally more urgent than a user with a stable score of 0.6, even though the latter exceeds the binary classification threshold. The risk scoring system captures this distinction.

#### Risk Tier Design

Four tiers were defined to map directly to governance response levels:

| Tier | Score Threshold | Governance Response |
|---|---|---|
| Critical | >= 0.75 | Immediate analyst review, access restriction consideration |
| High | >= 0.50 | Active monitoring, behavioral interview consideration |
| Medium | >= 0.25 | Passive monitoring, quarterly review |
| Low | < 0.25 | Normal operations, no additional monitoring required |

The thresholds were set to align with standard risk treatment levels in ISO 27001 Annex A and NIST SP 800-53 rather than purely statistical criteria.

#### Score Calibration

Tree-based models produce probability estimates that are not inherently calibrated — a score of 0.7 does not automatically mean a 70% historical likelihood of being malicious. For a system used in personnel risk decisions, calibration is a governance requirement under NIST AI RMF 1.0. Isotonic regression and Platt scaling were applied and evaluated using the Brier Score. Isotonic regression was selected as the production calibration method, achieving a 42% Brier score improvement over uncalibrated outputs.

#### Temporal Risk Trajectory

Monthly cumulative feature snapshots were computed across all 17 observation months for a sample of malicious and benign users. This produces a longitudinal risk score for each user, revealing the behavioral ramp-up pattern that precedes insider incidents. This is the "low-and-slow" escalation signature documented in the CERT insider threat literature: insiders do not jump immediately to maximum risk — they accumulate behavioral anomalies gradually over weeks to months before incident completion.

#### Risk Velocity

Month-over-month risk score change (velocity) was computed to provide an early warning signal independent of absolute score level. A user at score 0.35 with velocity +0.20 this month is more actionable than a user at score 0.55 with velocity 0.00, even though only the latter exceeds the High tier threshold. Velocity alerting at a +0.10 monthly threshold detected 18 high-velocity events, 15 of which belonged to malicious users.

#### Feature Stability and Policy-to-Feature Mapping

SHAP rank stability was validated across 5 CV folds. All 19 features show HIGH stability (standard deviation below 2.0 across folds), with the top 6 features showing zero variance. A policy-to-feature mapping documents which organizational policy changes (BYOD, remote work, DLP, shift work) would affect which features and whether model retraining would be required. This is a CMMC maturity requirement for ML-based security systems.

---

## Methodology

### Feature Set

19 behavioral and psychometric features inherited from the enhanced Insider Threat Detection model:

**Baseline features (15):** logon_count, logoff_count, after_hours_logon, unique_pcs, device_count, unique_devices, email_sent, avg_email_size, total_attachments, unique_recipients, openness, conscientiousness, extraversion, agreeableness, neuroticism

**Velocity features (4):** after_hours_ratio, unique_devices_per_day, logon_burst_ratio, email_burst_ratio

### Risk Scoring Pipeline

1. Compute `predict_proba[:, 1]` for all 1,000 users using scaler_v2 and rf_v2
2. Assign risk tiers based on score thresholds (0.25 / 0.50 / 0.75)
3. Compute lift per tier against population baseline (7% malicious rate)
4. Apply SHAP analysis at tier level and compute explainability gap
5. Generate ranked risk report with top 3 drivers per user
6. Apply isotonic calibration for audit-ready probability estimates
7. Compute monthly risk trajectories and velocity for tracked users
8. Validate feature rank stability across 5 CV folds

---

## Results

### Risk Score Separation

![Risk Score Distribution](images/Risk_Score_Distribution.png)

![Risk Score Density](images/Risk_Score_Density.png)

| Population | Mean Risk Score |
|---|---|
| Benign (930 users) | 0.0077 |
| Malicious (70 users) | 0.9715 |

Near-perfect separation between benign and malicious score distributions. Benign users cluster overwhelmingly near 0; malicious users cluster near 1.0.

### Risk Tier Classification with Lift

![Risk Tier Lift](images/Risk_Tier_Lift.png)

![Risk Tier Malicious Rate](images/Risk_Tier_Malicious_Rate.png)

![Risk Tier User Counts](images/Risk_Tier_User_Counts.png)

| Tier | Users | Malicious | Mal Rate | Lift | % of All Malicious |
|---|---|---|---|---|---|
| Critical | 70 | 69 | 98.6% | **14.1x** | 98.6% |
| High | 4 | 1 | 25.0% | 3.6x | 1.4% |
| Medium | 4 | 0 | 0.0% | 0.0x | 0.0% |
| Low | 922 | 0 | 0.0% | 0.0x | 0.0% |

**Critical tier lift of 14.1x** means a user in the Critical tier is 14.1 times more likely to be malicious than a random user from the population. An analyst reviewing only the Critical tier catches 98.6% of all malicious users while reviewing just 7% of the user population, reducing investigative workload by 93%.

### SHAP Tier Analysis and Explainability Gap

![SHAP Tier Drivers](images/SHAP_Tier_Drivers.png)

![SHAP Explainability Gap](images/SHAP_Explainability_Gap.png)

The Critical tier is dominated by unique_devices_per_day and logoff_count. The Explainability Gap analysis reveals that device_count is the primary differentiator between true insiders (SHAP 0.0645) and false alarms (SHAP 0.0248) in the Critical/High tiers. False alarms tend to be driven by logoff_count and after_hours_ratio rather than device activity — a pattern analysts can use to triage alerts more efficiently.

### Score Calibration

![Calibration Curve](images/Risk_Score_Calibration_Curve.png)

| Method | Brier Score | Improvement |
|---|---|---|
| Uncalibrated | 0.0038 | Reference |
| Isotonic | **0.0022** | **42%** |
| Platt Scaling | 0.0037 | 3% |

The calibration curve confirms that isotonic-calibrated scores track closely to the perfect diagonal, satisfying the governance audit requirement that a score of 0.7 corresponds to approximately 70% likelihood of being malicious.

### Temporal Risk Trajectory

![Risk Trajectory Malicious](images/Risk_Trajectory_Malicious.png)

![Risk Trajectory Benign](images/Risk_Trajectory_Benign.png)

All five tracked malicious users show a clear low-and-slow behavioral ramp-up, crossing the Critical threshold between mid-2010 and early 2011. Benign users remain near zero throughout the observation period, with one temporary spike that self-corrects — the behavioral signature of a false positive.

### Risk Velocity

![Risk Velocity Malicious](images/Risk_Velocity_Malicious.png)

![Risk Velocity Distribution](images/Risk_Velocity_Distribution.png)

| Metric | Value |
|---|---|
| Velocity alert threshold | +0.10 per month |
| High velocity events detected | 18 |
| Belonging to malicious users | 15 (83%) |
| Belonging to benign users | 3 (17%) |

Malicious users show velocity spikes concentrated in mid-2010 (July to October), the behavioral acceleration phase prior to incident completion. Benign velocity is tightly clustered near zero with only minor fluctuations.

### Feature Stability

![Feature Stability](images/Feature_Stability.png)

All 19 features show HIGH stability (std < 2.0 across all 5 CV folds). The top 6 features (unique_devices_per_day through logon_count) have std = 0.00 — identical rankings across all five folds. The model's risk drivers are reproducible properties of the detection system, not artifacts of any particular data partition.

**Policy-to-feature mapping:**

| Policy Change | Retraining Required |
|---|---|
| BYOD Policy Change | YES — High Impact |
| Remote Work / WFH Policy | YES — High Impact |
| Email DLP Policy | MONITOR |
| Shared Workstation Policy | YES — High Impact |
| USB/Removable Media Policy | YES — High Impact |
| Flex Hours / Shift Work Policy | YES — High Impact |

---

## Key Research Findings

**Finding 1: From binary detection to risk spectrum.** Binary classification answers "is this user malicious?" The risk scoring system answers "how risky is this user, why, and how is that risk changing?" The Critical tier concentrates 98.6% of true malicious users into 7% of the population, reducing analyst workload by 93% without sacrificing detection coverage.

**Finding 2: Calibrated scores satisfy governance audit requirements.** Isotonic calibration achieves a 42% Brier score improvement, producing risk scores that track closely to empirical malicious rates. A score of 0.7 corresponds to approximately 70% likelihood — a requirement for personnel risk decisions under NIST AI RMF 1.0.

**Finding 3: Temporal trajectory reveals the disgruntlement phase.** The low-and-slow behavioral ramp-up confirmed across all five tracked malicious users is consistent with documented insider threat escalation patterns. Risk scoring enables early intervention during the Medium-to-High transition rather than reactive response after Critical threshold crossing.

**Finding 4: Velocity alerting detects emerging threats earlier than threshold crossing.** 15 of 18 high-velocity events belong to malicious users. A user rising rapidly at score 0.35 is detectable before they cross any standard alert threshold, providing a forward-looking governance signal.

**Finding 5: Psychometric features remain negligible in risk scoring context.** All five Big Five personality scores contribute less than 3% of SHAP impact at every risk tier. The risk scoring system requires no psychometric data — behavioral telemetry already captured by organizational IT systems is sufficient.

---

## Deployment Risk Profiling

The risk scoring system provides three levels of governance action:

**Automated pre-screening:** All 1,000 users are scored continuously. The 922 Low-tier users (92.2% of population) require no analyst attention. Automated alerts are generated only for Critical and High tier users.

**Tiered analyst review:** Critical tier users (approximately 70 in this dataset) receive immediate review with SHAP-driven explanations identifying the top 3 behavioral drivers. High tier users (approximately 4 in this dataset) receive active monitoring. Medium tier users receive passive monitoring with quarterly review.

**Velocity-based early warning:** Users crossing the +0.10 monthly velocity threshold receive a specialized early-warning alert independent of their current tier, enabling intervention during the behavioral ramp-up phase before Critical threshold crossing.

**Governance controls for deployment:** See the Insider Threat Detection project for full governance, technical, and legal deployment controls. All controls documented there apply equally to this risk scoring extension.

---

## AI Governance Principles

**Risk-based approach alignment:** Four-tier segmentation maps directly to ISO 27001 risk treatment levels and NIST SP 800-37 continuous monitoring categories.

**Calibration for audit readiness:** Isotonic regression produces probability estimates that satisfy the NIST AI RMF 1.0 requirement for honest performance reporting in systems affecting personnel decisions.

**Explainability at individual level:** Every flagged user receives a SHAP-driven explanation identifying their top 3 behavioral drivers. Analysts see "Score: 0.92, driven by: unique_devices_per_day, logoff_count, email_burst_ratio" rather than a raw number.

**Explainability gap documentation:** The TP vs FP SHAP comparison equips analysts to distinguish genuine insider signals from false alarm patterns, directly addressing alert fatigue.

**Feature stability for model governance:** All 19 features maintain HIGH rank stability across CV folds. The policy-to-feature mapping documents retraining triggers aligned with CMMC maturity requirements.

**Governance alignment:**
- NIST SP 800-37: continuous monitoring, risk-based access control
- CMMC Access Control (AC) and Audit (AU) domains: privileged user monitoring
- ISO 27001 A.7: human resource security, tiered risk treatment
- NIST AI RMF 1.0: calibration, explainability, deployment risk documentation
- EO 14110: trustworthy AI for systems affecting personnel decisions

---

## Technical Specification

| Parameter | Value |
|---|---|
| Base model | Enhanced Random Forest (19 features, from Insider Threat Detection) |
| Risk score source | predict_proba[:, 1] for all 1,000 users |
| Tier thresholds | Critical 0.75, High 0.50, Medium 0.25, Low < 0.25 |
| Calibration method | Isotonic regression (42% Brier score improvement) |
| Brier score (calibrated) | 0.0022 |
| Critical tier lift | 14.1x population baseline |
| Critical tier malicious rate | 98.6% |
| Temporal window | 17 months (2010-01 to 2011-05) |
| Velocity threshold | +0.10 per month |
| High velocity events | 18 (83% malicious) |
| Feature stability | All 19 features HIGH (std < 2.0 across 5 CV folds) |
| Policy triggers | 5 of 6 scenarios require retraining |
| Hardware | Kaggle CPU (no GPU required) |

---

## Repository Structure

See [STRUCTURE.md](STRUCTURE.md) for the full file tree and execution order.

**Quick start:**
```bash
pip install -r requirements.txt
python master_training_script.py
```

Scripts 01-18 restore the enhanced detection model. Scripts 19-26 execute the risk scoring extension.

---

## Related Work

This project is part of a 7-project ML portfolio in human-factor risk analytics for cybersecurity governance:

1. [IT Project Risk Classification](https://github.com/GraceE-Dion/IT-Project-Risk-Classification)
2. [Insider Threat Detection — CERT r4.2](https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2)
3. **Insider Risk Scoring — CERT r4.2** *(this project)*
4. Network Intrusion Detection *(in development)*
5. Phishing URL Detection *(planned)*
6. Financial / Digital Banking ML Suite *(planned)*
7. Medical Diagnosis Classification *(planned)*
8. Manufacturing Defect Detection *(planned)*

Projects 2 and 3 are directly connected: Insider Threat Detection performs binary classification; this project transforms those outputs into a continuous risk quantification framework. Together they represent a complete human-factor behavioral analytics pipeline spanning detection and governance-aligned risk scoring.

---

## Author

**Grace Egbedion**
Technical Program Manager | Cybersecurity Governance Specialist | PhD Candidate
MTSU, Computational and Data Science

Certifications: PMP, SAFe, PSM I/II, PSPO, CompTIA Security+

Research: Human-factor risk analytics, AI governance, IT cybersecurity program management

Publications: 5 peer-reviewed publications (2024-2025) | 27+ citations | 4,500+ reads

[GitHub](https://github.com/GraceE-Dion) | [LinkedIn](https://www.linkedin.com/in/grace-egbedion/)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

**Dataset:** The CERT Insider Threat Dataset r4.2 is property of Carnegie Mellon University Software Engineering Institute. Dataset usage is subject to CMU SEI terms of use and is not covered by the MIT License above. Access the dataset at: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

## Citation

If you reference this work:
```
Egbedion, G. (2025). Insider Risk Scoring: Continuous Behavioral Risk Quantification
for Cybersecurity Governance using CERT r4.2. GitHub Repository.
https://github.com/GraceE-Dion/Insider-Risk-Scoring-CERT-r4.2
```
