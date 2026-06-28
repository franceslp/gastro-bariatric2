#!/usr/bin/env python3
"""
build_matched_pairs_detail.py

Creates a side-by-side matched-pairs detail file: each row is one matched pair
with the GP patient's and their comparator's covariate values next to each other,
so you can eyeball match quality on the actual variables (not just PS distance).

Builds for both cohorts (with-BMI and no-BMI).
"""
import pandas as pd
import numpy as np

# Key covariates to show side by side (the clinically meaningful ones)
SHOW_COVS = [
    "age_at_surgery_approx", "baseline_a1c", "preoperative_bmi",
    "diabetes_duration_log1p", "t1dm", "t2dm",
    "hypertension", "ckd", "cad", "metformin", "any_insulin",
    "sex_encoded", "race_white", "race_black", "ethnicity_hispanic",
    "sleeve_vs_bypass",
]

COHORTS = {
    "with_BMI": ("psm_matched_pairs_new.csv",    "psm_full_covariate_matrix.csv",
                 "matched_pairs_detail_with_BMI.csv"),
    "no_BMI":   ("psm_matched_pairs_no_BMI.csv", "psm_full_covariate_matrix_no_BMI.csv",
                 "matched_pairs_detail_no_BMI.csv"),
}

for cname, (pairs_f, matrix_f, out_f) in COHORTS.items():
    pairs = pd.read_csv(pairs_f, dtype={"gp_patient_id": str, "comp_patient_id": str})
    matrix = pd.read_csv(matrix_f, dtype={"patient_id": str})

    cov_cols = [c for c in SHOW_COVS if c in matrix.columns]
    cov = matrix[["patient_id"] + cov_cols].copy()

    # GP side
    gp = cov.add_prefix("gp_").rename(columns={"gp_patient_id": "gp_patient_id"})
    # Comparator side
    co = cov.add_prefix("comp_").rename(columns={"comp_patient_id": "comp_patient_id"})

    detail = pairs.merge(gp, on="gp_patient_id", how="left")
    detail = detail.merge(co, on="comp_patient_id", how="left")

    # Reorder: pair info, then interleave gp_/comp_ for each covariate
    ordered = ["gp_patient_id", "comp_patient_id", "ps_distance"]
    for c in cov_cols:
        ordered += [f"gp_{c}", f"comp_{c}"]
    ordered = [c for c in ordered if c in detail.columns]
    detail = detail[ordered]

    detail.to_csv(out_f, index=False)
    print(f"{cname}: wrote {out_f} ({len(detail)} pairs, {len(detail.columns)} cols)")

    # Quick quality metric: mean absolute difference on continuous covars
    for c in ["age_at_surgery_approx", "baseline_a1c", "preoperative_bmi"]:
        if f"gp_{c}" in detail.columns and f"comp_{c}" in detail.columns:
            g = pd.to_numeric(detail[f"gp_{c}"], errors="coerce")
            cc = pd.to_numeric(detail[f"comp_{c}"], errors="coerce")
            mad = (g - cc).abs().mean()
            print(f"    mean |GP - comp| {c}: {mad:.2f}")

print("\nDone.")
