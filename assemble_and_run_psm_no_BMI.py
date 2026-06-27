"""
assemble_and_run_psm_no_BMI.py  — SENSITIVITY ANALYSIS (BMI EXCLUDED FROM MATCHING)

This is the NO-BMI sensitivity version. preoperative_bmi is deliberately
REMOVED from PSM_COVARIATES so that patients missing BMI are retained
(complete-case analysis does not drop them for missing BMI). BMI columns are
still loaded but NOT used in the propensity model. Compare against the primary
with-BMI analysis (assemble_and_run_psm.py) to assess robustness to BMI handling.
Outputs carry the _no_BMI suffix.
=====================================================================
assemble_and_run_psm.py

Assembles the full 31-covariate PSM-ready dataset from all collected
covariate files, then runs 1:1 nearest-neighbor propensity score matching.

Methodology (per Sadda et al. JAMA Surgery 2026):
  - Propensity score estimated via logistic regression on 31 covariates
  - 1:1 nearest-neighbor greedy matching
  - Caliper: 0.2 SD of logit propensity score (Austin 2011)
  - Balance assessed via standardized mean differences (SMD < 0.1)

INPUTS (GP cohort):
  cohort_FINAL_analytic.csv
  study_covariates_new.csv
  psm_covariates_true_diabetes_duration.csv

INPUTS (Comparator):
  comparator_pool_ready_for_PSM.csv
  comparator_pool_ready_for_PSM_with_BMI.csv
  psm_covariates_comorbidities.csv
  psm_covariates_labs.csv
  psm_covariates_comparator_meds_dx.csv
  psm_covariates_comparator_demographics.csv
  psm_covariates_true_diabetes_duration.csv

OUTPUTS:
  psm_full_covariate_matrix.csv   — all patients, all 31 covariates, pre-match
  psm_smd_before_matching.csv     — SMD table before matching
  psm_matched_pairs_new.csv       — matched patient_id pairs
  psm_matched_dataset_new.csv     — full covariate data for matched patients
  psm_smd_after_matching.csv      — SMD table after matching
  psm_balance_table_new.csv       — combined before/after balance table
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

print(">>> SCRIPT VERSION: assemble_and_run_psm_NO_BMI <<<")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Patient IDs to exclude from GP cohort (pending PI confirmation)
# Set to empty list [] if PI says keep them
EXCLUDE_GP_AMBIGUOUS_CPT = []  # exclusions already applied in step5_multisurgery_exclusion.py

# Comparator patients with post-surgical diabetes (invalid — exclude always)
INVALID_COMP_FILE = "psm_covariates_true_diabetes_duration.csv"

CALIPER = 0.2   # SD of logit PS — per Austin 2011 / Sadda

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smd(x1, x2):
    """Standardized mean difference for a single covariate."""
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(), x2.var()
    pooled_sd = np.sqrt((v1 + v2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(m1 - m2) / pooled_sd

def smd_table(gp_df, comp_df, covariates):
    """Compute SMD for all covariates between two groups."""
    rows = []
    for cov in covariates:
        if cov in gp_df.columns and cov in comp_df.columns:
            g = pd.to_numeric(gp_df[cov], errors="coerce").dropna()
            c = pd.to_numeric(comp_df[cov], errors="coerce").dropna()
            s = smd(g, c)
            rows.append({
                "covariate": cov,
                "gp_mean":   round(g.mean(), 4),
                "comp_mean": round(c.mean(), 4),
                "smd":       round(s, 4),
                "balanced":  s < 0.1,
            })
    return pd.DataFrame(rows)

# ===========================================================================
# SECTION 1 — BUILD GP COVARIATE MATRIX
# ===========================================================================
print("\n--- Building GP covariate matrix ---")

gp_base = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
gp_cov  = pd.read_csv("study_covariates_new.csv", dtype={"patient_id": str})
gp_dur  = pd.read_csv("psm_covariates_true_diabetes_duration.csv",
                      dtype={"patient_id": str})

# Exclude ambiguous CPT patients (pending PI confirmation)
if EXCLUDE_GP_AMBIGUOUS_CPT:
    before = len(gp_base)
    gp_base = gp_base[~gp_base["patient_id"].isin(EXCLUDE_GP_AMBIGUOUS_CPT)]
    print(f"Excluded {before - len(gp_base)} ambiguous CPT patients "
          f"→ GP n={len(gp_base)}")

# Encode demographics from funnel file
race_lower = gp_base["race"].str.strip().str.lower().fillna("")
eth_lower  = gp_base["ethnicity"].str.strip().str.lower().fillna("")
gp_base["race_white"]         = race_lower.str.contains("white|caucasian").astype(int)
gp_base["race_black"]         = race_lower.str.contains("black|african").astype(int)
gp_base["ethnicity_hispanic"] = eth_lower.str.contains("hispanic|latino|spanish").astype(int)

# sleeve_vs_bypass: bypass takes precedence when both codes present
# 43775 only → 1 (sleeve)
# any bypass code (43644/43645/43846/43847), with or without 43775 → 0 (bypass)
# Bypass precedence: any bypass code wins over sleeve. Ambiguous patients already excluded in step5_multisurgery_exclusion.py (EXCLUDE_GP_AMBIGUOUS_CPT=[])
cpt = gp_base["bariatric_cpt_codes_seen"].astype(str).str.replace(r"\.0","",regex=True)
gp_base["sleeve_vs_bypass"] = np.where(
    cpt.str.contains("43644|43645|43846|43847", regex=True, na=False),
    0,   # bypass (including mixed codes)
    np.where(
        cpt.str.contains("43775", na=False),
        1,   # sleeve only
        np.nan  # unknown
    )
)

# Sex encode (F=1, M=0)
gp_base["sex_encoded"] = (gp_base["sex"].str.upper() == "F").astype(int)

# Surgery year
gp_base["surgery_year"] = pd.to_datetime(
    gp_base["bariatric_date"], errors="coerce"
).dt.year

# Merge all GP sources
gp = gp_base[["patient_id", "age_at_surgery_approx", "sex_encoded",
               "race_white", "race_black", "ethnicity_hispanic",
               "sleeve_vs_bypass", "surgery_year"]].copy()
gp = gp.merge(gp_cov[[
    "patient_id", "baseline_a1c",
    "t1dm", "t2dm",
    "dm_renal", "dm_neuro", "dm_circ", "dm_opthal", "dm_other",
    "hypertension", "ckd", "cad", "stroke", "heart_failure", "dyslipidemia",
    "metformin", "any_insulin", "rapid_insulin", "long_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
]], on="patient_id", how="left")

# GP BMI from vitals-based file (study_covariates_new.csv BMI is 100% missing —
# it scanned lab_result.csv instead of vitals_signs.csv. The dedicated BMI
# script find_BMI_at_or_before_surgery.py correctly reads vitals.)
gp_bmi = pd.read_csv("gastroparesis_cohort_BMI_at_or_before_surgery.csv",
                     dtype={"patient_id": str},
                     usecols=["patient_id", "BMI_at_or_before_surgery"])
gp_bmi = gp_bmi.rename(columns={"BMI_at_or_before_surgery": "preoperative_bmi"})
gp = gp.merge(gp_bmi, on="patient_id", how="left")

gp_dur_gp = gp_dur[gp_dur["group"] == "gastroparesis"][
    ["patient_id", "diabetes_duration_days", "diabetes_duration_log1p",
     "diabetes_duration_winsorized_days"]
]
gp = gp.merge(gp_dur_gp, on="patient_id", how="left")
gp["group"] = "gastroparesis"

print(f"GP matrix: {len(gp)} patients, {len(gp.columns)} columns")

# ===========================================================================
# SECTION 2 — BUILD COMPARATOR COVARIATE MATRIX
# ===========================================================================
print("\n--- Building comparator covariate matrix ---")

comp_base  = pd.read_csv("comparator_pool_ready_for_PSM.csv",
                         dtype={"patient_id": str})
comp_bmi   = pd.read_csv("comparator_pool_ready_for_PSM_with_BMI.csv",
                         dtype={"patient_id": str})
comp_comorb= pd.read_csv("psm_covariates_comorbidities.csv",
                         dtype={"patient_id": str})
comp_labs  = pd.read_csv("psm_covariates_labs.csv",
                         dtype={"patient_id": str})
comp_meds  = pd.read_csv("psm_covariates_comparator_meds_dx.csv",
                         dtype={"patient_id": str})
comp_demo  = pd.read_csv("psm_covariates_comparator_demographics.csv",
                         dtype={"patient_id": str})
comp_dur   = gp_dur[gp_dur["group"] == "comparator"][
    ["patient_id", "diabetes_duration_days", "diabetes_duration_log1p",
     "diabetes_duration_winsorized_days"]
]

# Remove invalid comparators with diabetes diagnosed only after surgery (negative duration)
dur_all = pd.read_csv(INVALID_COMP_FILE, dtype={"patient_id": str})
invalid_comp = set(
    dur_all.loc[
        (dur_all["group"] == "comparator") &
        (dur_all["diabetes_duration_days"] < 0),
        "patient_id"
    ]
)
before = len(comp_base)
comp_base = comp_base[~comp_base["patient_id"].isin(invalid_comp)]
print(f"Removed {before - len(comp_base)} invalid comparators "
      f"(post-surgical diabetes) → n={len(comp_base)}")

# Encode comparator demographics
# sex was not collected by collect_comparator_demographics.py; pull it from
# the dedicated comparator_sex.csv (collect_comparator_sex.py scans patient.csv)
comp_sex = pd.read_csv("comparator_sex.csv", dtype={"patient_id": str})
comp_base = comp_base.merge(comp_sex[["patient_id", "sex_encoded"]],
                            on="patient_id", how="left")
comp_base["surgery_year"] = pd.to_datetime(
    comp_base["bariatric_date"], errors="coerce"
).dt.year

# sleeve_vs_bypass from surgery_type column
comp_base["sleeve_vs_bypass"] = (
    comp_base["surgery_type"].str.lower().str.contains("sleeve", na=False)
).astype(int)

# Build comparator matrix
comp = comp_base[["patient_id", "age_at_surgery_approx", "sex_encoded",
                   "sleeve_vs_bypass", "surgery_year"]].copy()

# Demographics
comp = comp.merge(
    comp_demo[["patient_id", "race_white", "race_black",
               "ethnicity_hispanic", "t1dm", "t2dm"]],
    on="patient_id", how="left"
)

# Labs — rename to match GP column names
comp_labs_r = comp_labs[["patient_id", "a1c_baseline"]].rename(
    columns={"a1c_baseline": "baseline_a1c"}
)
comp = comp.merge(comp_labs_r, on="patient_id", how="left")

# BMI
comp_bmi_r = comp_bmi[["patient_id", "BMI_at_or_before_surgery"]].rename(
    columns={"BMI_at_or_before_surgery": "preoperative_bmi"}
)
comp = comp.merge(comp_bmi_r, on="patient_id", how="left")

# Comorbidities — rename to match GP
comp_comorb_r = comp_comorb[[
    "patient_id", "hypertension", "ckd", "cad", "stroke", "heart_failure",
]].copy()
# diabetic_nephropathy → dm_renal, diabetic_neuropathy → dm_neuro,
# diabetic_retinopathy → dm_opthal
comp_comorb_r2 = comp_comorb[["patient_id",
    "diabetic_nephropathy", "diabetic_neuropathy", "diabetic_retinopathy"
]].rename(columns={
    "diabetic_nephropathy": "dm_renal",
    "diabetic_neuropathy":  "dm_neuro",
    "diabetic_retinopathy": "dm_opthal",
})
comp = comp.merge(comp_comorb_r,  on="patient_id", how="left")
comp = comp.merge(comp_comorb_r2, on="patient_id", how="left")

# Medications + dm_circ, dm_other, dyslipidemia
comp_meds_cols = ["patient_id", "metformin", "any_insulin", "rapid_insulin",
                  "long_insulin", "glp1", "sglt2", "dpp4", "sulfonylurea",
                  "tzd", "dm_circulatory", "dm_other", "dyslipidemia"]
comp_meds_r = comp_meds[[c for c in comp_meds_cols if c in comp_meds.columns]].copy()
comp_meds_r = comp_meds_r.rename(columns={"dm_circulatory": "dm_circ"})
comp = comp.merge(comp_meds_r, on="patient_id", how="left")

# Diabetes duration
comp = comp.merge(comp_dur, on="patient_id", how="left")
comp["group"] = "comparator"

print(f"Comparator matrix: {len(comp)} patients, {len(comp.columns)} columns")

# ===========================================================================
# SECTION 3 — STACK AND VALIDATE
# ===========================================================================
print("\n--- Stacking GP + comparator ---")

# Align columns
all_cols = list(dict.fromkeys(list(gp.columns) + list(comp.columns)))
for df in [gp, comp]:
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan

psm_df = pd.concat([gp[all_cols], comp[all_cols]], ignore_index=True)
psm_df["group_encoded"] = (psm_df["group"] == "gastroparesis").astype(int)

print(f"Combined matrix: {len(psm_df)} patients")
print(f"  GP:         {(psm_df['group']=='gastroparesis').sum()}")
print(f"  Comparator: {(psm_df['group']=='comparator').sum()}")

# ===========================================================================
# SECTION 4 — PSM COVARIATES AND MISSINGNESS
# ===========================================================================
PSM_COVARIATES = [
    "age_at_surgery_approx", "baseline_a1c",
    "diabetes_duration_log1p",
    "t1dm", "t2dm",
    "dm_renal", "dm_neuro", "dm_circ", "dm_opthal", "dm_other",
    "hypertension", "ckd", "cad", "stroke", "heart_failure", "dyslipidemia",
    "metformin", "any_insulin", "rapid_insulin", "long_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
    "sex_encoded", "race_white", "race_black", "ethnicity_hispanic",
    "sleeve_vs_bypass",
]

gp_psm   = psm_df[psm_df["group"] == "gastroparesis"]
comp_psm = psm_df[psm_df["group"] == "comparator"]

print(f"\nMissingness report — NO IMPUTATION (complete case analysis):")
print(f"  {'Covariate':<30} {'Overall':>10} {'GP':>10} {'Comparator':>12} {'Diff':>8}")
print("  " + "-"*74)
for col in PSM_COVARIATES:
    if col in psm_df.columns:
        overall_pct = psm_df[col].isna().mean() * 100
        gp_pct      = gp_psm[col].isna().mean() * 100
        comp_pct    = comp_psm[col].isna().mean() * 100
        diff        = abs(gp_pct - comp_pct)
        flag        = " ⚠ differential" if diff > 10 else ""
        print(f"  {col:<30} {overall_pct:>9.1f}% {gp_pct:>9.1f}% "
              f"{comp_pct:>11.1f}% {diff:>7.1f}%{flag}")

# ===========================================================================
# SECTION 5 — COMPLETE CASE SELECTION (no imputation)
# ===========================================================================
print("\n--- Complete case selection (no imputation) ---")
model_df = psm_df[["patient_id", "group", "group_encoded"] + PSM_COVARIATES].copy()
model_df = model_df.dropna(subset=PSM_COVARIATES)

n_dropped    = len(psm_df) - len(model_df)
n_dropped_gp = len(gp_psm) - (model_df["group"] == "gastroparesis").sum()
n_dropped_co = len(comp_psm) - (model_df["group"] == "comparator").sum()

print(f"Total dropped (missing any covariate): {n_dropped} "
      f"({n_dropped/len(psm_df)*100:.1f}%)")
print(f"  GP dropped:         {n_dropped_gp} "
      f"({n_dropped_gp/len(gp_psm)*100:.1f}%)")
print(f"  Comparator dropped: {n_dropped_co} "
      f"({n_dropped_co/len(comp_psm)*100:.1f}%)")
print(f"Complete cases: {len(model_df)} "
      f"(GP={( model_df['group']=='gastroparesis').sum()}, "
      f"Comparator={(model_df['group']=='comparator').sum()})")
print("NOTE: Patients with missing values for any prespecified covariate "
      "were excluded from propensity score estimation (no imputation).")

# ===========================================================================
# SECTION 6 — SMD BEFORE MATCHING (on complete cases — same population as model)
# ===========================================================================
# Important: SMD computed on complete-case population, not all patients,
# so before/after SMD tables represent the same population
gp_cc   = model_df[model_df["group"] == "gastroparesis"]
comp_cc = model_df[model_df["group"] == "comparator"]

smd_before = smd_table(gp_cc, comp_cc, PSM_COVARIATES)
print(f"\nSMD before matching (complete cases only):")
print(f"  Balanced (SMD<0.1): {smd_before['balanced'].sum()}/{len(smd_before)}")
print(f"  Max SMD: {smd_before['smd'].max():.3f} "
      f"({smd_before.loc[smd_before['smd'].idxmax(),'covariate']})")
smd_before.to_csv("psm_smd_before_matching_no_BMI.csv", index=False)

# ===========================================================================
# SECTION 7 — PROPENSITY SCORE MODEL (logistic regression, complete cases)
# ===========================================================================
print("\n--- Fitting propensity score model ---")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = model_df[PSM_COVARIATES].values
y = model_df["group_encoded"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

model_df["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]
model_df["logit_ps"] = np.log(
    model_df["propensity_score"] / (1 - model_df["propensity_score"])
)

print(f"Model converged: {lr.n_iter_[0]} iterations")
print(f"PS range — GP:   {model_df.loc[model_df['group_encoded']==1,'propensity_score'].describe().round(3).to_string()}")

# ===========================================================================
# SECTION 7 — 1:1 NEAREST NEIGHBOR MATCHING WITH CALIPER
# ===========================================================================
print(f"\n--- 1:1 nearest-neighbor matching (caliper={CALIPER} SD logit PS) ---")

logit_sd = model_df["logit_ps"].std()
caliper_val = CALIPER * logit_sd
print(f"Logit PS SD: {logit_sd:.4f} | Caliper: {caliper_val:.4f}")

gp_m    = model_df[model_df["group_encoded"] == 1].copy().reset_index(drop=True)
comp_m  = model_df[model_df["group_encoded"] == 0].copy().reset_index(drop=True)

# Build KDTree on comparator logit PS for fast nearest-neighbor search
tree = KDTree(comp_m[["logit_ps"]].values)

matched_pairs = []
used_comp = set()

# Sort GP patients by propensity score for deterministic greedy matching
# (ascending PS → closest to comparator distribution matched first)
gp_sorted = gp_m.sort_values("propensity_score").reset_index(drop=True)

for _, gp_row in gp_sorted.iterrows():
    gp_logit = gp_row["logit_ps"]
    # Find nearest comparator within caliper
    dists, idxs = tree.query([[gp_logit]], k=min(10, len(comp_m)))
    for dist, idx in zip(dists[0], idxs[0]):
        if dist <= caliper_val and idx not in used_comp:
            matched_pairs.append({
                "gp_patient_id":   gp_row["patient_id"],
                "comp_patient_id": comp_m.loc[idx, "patient_id"],
                "gp_logit_ps":     gp_logit,
                "comp_logit_ps":   comp_m.loc[idx, "logit_ps"],
                "ps_distance":     dist,
            })
            used_comp.add(idx)
            break

pairs_df = pd.DataFrame(matched_pairs)
print(f"Matched pairs: {len(pairs_df)} of {len(gp_m)} GP patients")
print(f"Unmatched GP:  {len(gp_m) - len(pairs_df)} "
      f"(outside caliper or no available comparator)")

# ===========================================================================
# SECTION 8 — POST-MATCH BALANCE
# ===========================================================================
matched_gp_ids   = set(pairs_df["gp_patient_id"])
matched_comp_ids = set(pairs_df["comp_patient_id"])

matched_gp   = psm_df[psm_df["patient_id"].isin(matched_gp_ids)]
matched_comp = psm_df[psm_df["patient_id"].isin(matched_comp_ids)]

smd_after = smd_table(matched_gp, matched_comp, PSM_COVARIATES)
smd_after = smd_after.rename(columns={
    "gp_mean": "gp_mean_after", "comp_mean": "comp_mean_after",
    "smd": "smd_after", "balanced": "balanced_after"
})

balance = smd_before.merge(
    smd_after[["covariate","gp_mean_after","comp_mean_after",
               "smd_after","balanced_after"]],
    on="covariate"
)

print(f"\nBalance after matching:")
print(f"  Balanced (SMD<0.1): {smd_after['balanced_after'].sum()}/{len(smd_after)}")
print(f"  Max SMD after:      {smd_after['smd_after'].max():.3f}")
print(f"\n  {'Covariate':<30} {'SMD before':>12} {'SMD after':>10} {'Balanced':>10}")
print("  " + "-"*66)
for _, row in balance.iterrows():
    flag = "✓" if row["balanced_after"] else "⚠"
    print(f"  {row['covariate']:<30} {row['smd']:>12.3f} "
          f"{row['smd_after']:>10.3f} {flag:>10}")

# ===========================================================================
# SECTION 9 — SAVE OUTPUTS
# ===========================================================================
psm_df.to_csv("psm_full_covariate_matrix_no_BMI.csv", index=False)
pairs_df.to_csv("psm_matched_pairs_no_BMI.csv", index=False)
balance.to_csv("psm_balance_table_no_BMI.csv", index=False)
smd_before.to_csv("psm_smd_before_matching_no_BMI.csv", index=False)
smd_after.to_csv("psm_smd_after_matching_no_BMI.csv", index=False)

# Full matched dataset
matched_dataset = psm_df[
    psm_df["patient_id"].isin(matched_gp_ids | matched_comp_ids)
].copy()
matched_dataset.to_csv("psm_matched_dataset_no_BMI.csv", index=False)

print(f"\nOutputs written:")
print(f"  psm_full_covariate_matrix.csv   ({len(psm_df):,} rows)")
print(f"  psm_matched_pairs_new.csv        ({len(pairs_df):,} pairs)")
print(f"  psm_matched_dataset_new.csv      ({len(matched_dataset):,} rows)")
print(f"  psm_balance_table_new.csv        ({len(balance):,} covariates)")
print(f"  psm_smd_before_matching.csv")
print(f"  psm_smd_after_matching.csv")

assert pairs_df["gp_patient_id"].nunique() == len(pairs_df), \
    "Duplicate GP patients in matched pairs"
assert pairs_df["comp_patient_id"].nunique() == len(pairs_df), \
    "Duplicate comparator patients in matched pairs"
print("\nAll assertions passed ✓")
print("Done.")
