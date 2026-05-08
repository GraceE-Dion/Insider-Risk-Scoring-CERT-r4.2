# =============================================================================
# INHERITED_SCRIPTS_NOTE.md
# Insider Risk Scoring — CERT r4.2
#
# Scripts 02 through 07 (data loading, feature engineering, merge and label,
# preprocessing, model training, evaluation) are inherited directly from the
# Insider Threat Detection project:
# https://github.com/GraceE-Dion/Insider-Threat-Detection-CERT-r4.2
#
# Scripts 08 through 18 (SHAP, threshold tuning, cross-validation, save
# outputs, PR-AUC, cost matrix, user-ID verification, sensitivity test,
# velocity features, enhanced model, enhanced SHAP) are also inherited.
#
# This repository adds Scripts 19 through 26 which implement the continuous
# risk scoring extension on top of the enhanced 19-feature model (rf_v2).
#
# Run order:
#   01 through 18  — inherited from Insider Threat Detection repo
#   19 through 26  — new risk scoring pipeline (this repo)
# =============================================================================
