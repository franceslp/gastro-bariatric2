#!/usr/bin/env python3
"""
build_optimal_balance_full.py

Generates the complete pre+post balance table for the OPTIMAL matched cohort,
in the same shape as the original psm_balance_table_new.csv (used to build
the "PSM Balance (with-BMI)" sheet):
  covariate | gp_mean_pre | comp_mean_pre | smd_pre | balanced_pre |
  gp_mean_post | comp_mean_post | smd_post | balanced_post

PRE-match: computed over ALL complete-case patients (229 GP vs 2,797
comparator pool) — same population as the original pre-match SMD, i.e.
before any matching was applied.

POST-match: computed over the 227 optimal-matched pairs.

Reads only existing files. Installs nothing.
Output: optimal_balance_full.csv
"""
import numpy as np
import pandas as pd

MATRIX = "psm_full_covariate_matrix.csv"
PAIRS  = "optimal_logit_caliper_pairs.csv"

COVARIATES = [
    "age_at_surgery_approx", "baseline_a1c", "preoperative_bmi",
    "diabetes_duration_log1p",
    "t1dm", "t2dm",
    "dm_renal", "dm_neuro", "dm_circ", "dm_opthal", "dm_other",
    "hypertension", "ckd", "cad", "stroke", "heart_failure", "dyslipidemia",
    "metformin", "any_insulin", "rapid_insulin", "long_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
    "sex_encoded", "race_white", "race_black", "ethnicity_hispanic",
    "sleeve_vs_bypass",
]

def smd(a, b):
    # drop NaN independently in each arm (matches complete-case convention;
    # guards against a stray missing sleeve_vs_bypass poisoning the row)
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else abs(a.mean() - b.mean()) / pooled

mat = pd.read_csv(MATRIX, dtype={"patient_id": str})
cc = mat.dropna(subset=COVARIATES + ["group_encoded"]).reset_index(drop=True)

gp_pre   = cc[cc["group_encoded"] == 1]
comp_pre = cc[cc["group_encoded"] == 0]
print(f"Pre-match (complete-case): GP={len(gp_pre)}, Comparator={len(comp_pre)}")

pairs = pd.read_csv(PAIRS, dtype={"gp_id": str, "comp_id": str})
# Pull post-match patients from the COMPLETE-CASE set (cc), not the raw matrix,
# so the post-match balance is provably computed on the exact population that
# matching drew from — guards against duplicate/modified rows in the raw matrix.
gp_post   = cc[cc["patient_id"].isin(pairs["gp_id"])]
comp_post = cc[cc["patient_id"].isin(pairs["comp_id"])]
print(f"Post-match (optimal): GP={len(gp_post)}, Comparator={len(comp_post)}")
assert len(gp_post) == len(pairs), "GP post-match count mismatch"
assert len(comp_post) == len(pairs), "Comparator post-match count mismatch"
assert gp_post["patient_id"].is_unique, "duplicate GP patient_ids in matched set"
assert comp_post["patient_id"].is_unique, "duplicate comparator patient_ids in matched set"

rows = []
for cov in COVARIATES:
    gp_mean_pre   = pd.to_numeric(gp_pre[cov], errors="coerce").mean()
    comp_mean_pre = pd.to_numeric(comp_pre[cov], errors="coerce").mean()
    smd_pre_val   = smd(gp_pre[cov], comp_pre[cov])

    gp_mean_post   = pd.to_numeric(gp_post[cov], errors="coerce").mean()
    comp_mean_post = pd.to_numeric(comp_post[cov], errors="coerce").mean()
    smd_post_val   = smd(gp_post[cov], comp_post[cov])

    if np.isnan(smd_pre_val) or np.isnan(smd_post_val):
        print(f"  WARNING: NaN SMD for '{cov}' — check for missing values")

    rows.append({
        "covariate": cov,
        "gp_mean_pre": round(gp_mean_pre, 4),
        "comp_mean_pre": round(comp_mean_pre, 4),
        "smd_pre": round(smd_pre_val, 4),
        "balanced_pre": bool(smd_pre_val < 0.1) if not np.isnan(smd_pre_val) else None,
        "gp_mean_post": round(gp_mean_post, 4),
        "comp_mean_post": round(comp_mean_post, 4),
        "smd_post": round(smd_post_val, 4),
        "balanced_post": bool(smd_post_val < 0.1) if not np.isnan(smd_post_val) else None,
    })

bal = pd.DataFrame(rows)
bal.to_csv("optimal_balance_full.csv", index=False)

# fail loudly if any SMD came out NaN (broken row rather than real result)
if bal[["smd_pre", "smd_post"]].isna().any().any():
    print("\n*** WARNING: some SMDs are NaN — do NOT use this table until resolved ***")

n_bal_pre  = bal["balanced_pre"].sum()
n_bal_post = bal["balanced_post"].sum()
print(f"\nBalanced pre-match:  {n_bal_pre}/{len(bal)}")
print(f"Balanced post-match: {n_bal_post}/{len(bal)}")
print(f"\nWrote optimal_balance_full.csv")
print(bal.to_string(index=False))
