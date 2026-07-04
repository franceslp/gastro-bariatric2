#!/usr/bin/env python3
"""
build_optimal_matched_dataset.py

Bridges the new optimal-matched cohort into the existing A1c trajectory
pipeline. The optimal matching (optimal_logit_caliper.py) saved only the pairs
(optimal_logit_caliper_pairs.csv). The trajectory builder
(collect_a1c_trajectory.py) needs a MATCHED DATASET file with patient_id,
group, and baseline_a1c for the matched patients.

This script:
  1. Reads the 227 optimal pairs (gp_id, comp_id)
  2. Pulls group + baseline_a1c (+ all covariates) for those patients from
     psm_full_covariate_matrix.csv
  3. Writes psm_matched_dataset_optimal.csv  (same format as the greedy
     psm_matched_dataset_new.csv, so the trajectory builder can consume it)
  4. Also writes psm_matched_pairs_optimal.csv with the gp_patient_id /
     comp_patient_id column names the builder's cross-check expects.

After running this, regenerate the trajectory with the OPTIMAL cohort by
running collect_a1c_trajectory_optimal.py (provided separately), then
analyze_a1c_trajectory.py unchanged.
"""
import pandas as pd

PAIRS_IN   = "optimal_logit_caliper_pairs.csv"       # gp_id, comp_id, ...
MATRIX     = "psm_full_covariate_matrix.csv"          # all patients + covariates + group + baseline_a1c
DATASET_OUT = "psm_matched_dataset_optimal.csv"
PAIRS_OUT   = "psm_matched_pairs_optimal.csv"

pairs = pd.read_csv(PAIRS_IN, dtype={"gp_id": str, "comp_id": str})
matrix = pd.read_csv(MATRIX, dtype={"patient_id": str})

print(f"Optimal pairs: {len(pairs)}")
print(f"Full matrix:   {len(matrix)} patients")

# --- sanity: matrix must have the columns the trajectory builder needs ---
need = {"patient_id", "group", "baseline_a1c"}
missing = need - set(matrix.columns)
if missing:
    raise SystemExit(f"Matrix missing required columns: {missing}")

# --- collect the matched patient IDs (GP + comparator) ---
gp_ids   = pairs["gp_id"].tolist()
comp_ids = pairs["comp_id"].tolist()
all_ids  = gp_ids + comp_ids

# guard against duplicates (optimal 1:1 should have none)
assert len(set(gp_ids)) == len(gp_ids),   "duplicate GP ids in optimal pairs"
assert len(set(comp_ids)) == len(comp_ids), "duplicate comparator ids in optimal pairs"

# --- build matched dataset in the same shape as psm_matched_dataset_new.csv ---
matched = matrix[matrix["patient_id"].isin(all_ids)].copy()
print(f"Matched patients pulled from matrix: {len(matched)} "
      f"(expected {len(all_ids)})")

# every matched patient must be found in the matrix
found = set(matched["patient_id"])
missing_ids = set(all_ids) - found
if missing_ids:
    raise SystemExit(f"{len(missing_ids)} matched patients not found in matrix "
                     f"(id format mismatch?): e.g. {list(missing_ids)[:5]}")

# baseline_a1c coverage (should be ~100% — A1c was a required PSM covariate)
n_base = matched["baseline_a1c"].notna().sum()
print(f"baseline_a1c present: {n_base}/{len(matched)} "
      f"({100*n_base/len(matched):.1f}%)")

matched.to_csv(DATASET_OUT, index=False)
print(f"wrote {DATASET_OUT}")

# --- pairs file with the column names the builder expects ---
pairs_out = pairs.rename(columns={"gp_id": "gp_patient_id",
                                  "comp_id": "comp_patient_id"})
pairs_out.to_csv(PAIRS_OUT, index=False)
print(f"wrote {PAIRS_OUT}")

# --- report group breakdown ---
print("\nGroup breakdown in matched dataset:")
print(matched["group"].value_counts().to_string())
print("\nDone. Next: run collect_a1c_trajectory_optimal.py")
