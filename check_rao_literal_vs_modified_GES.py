"""
check_rao_literal_vs_modified_GES.py

Quick follow-up - no rescan needed. The diagnosis.csv/procedure.csv scans
for symptoms and upper endoscopy are already done and saved in the Rao-
adapted output file. Only the GES timing window needs recomputing, since
that's pure arithmetic on first_GES_date vs first_K31_84_date, both
already in the file.

Computes TWO versions side by side:
  - LITERAL Rao et al. definition: GES 7 to 90 days BEFORE dx (their
    actual published "1 week to 3 months" window)
  - MODIFIED version (already in the file as ges_within_90d_before_dx):
    GES 0 to 90 days before dx, same-day included

Both "meets all three criteria" flags use the SAME symptom and endoscopy
columns already computed - only the GES piece differs between them. This
makes the two "meets criteria" numbers a clean, isolated comparison of
just that one design choice.

Run on the VM - instant, no GCS, no nohup needed.
"""

import pandas as pd

INPUT_CSV = "gastroparesis_prokinetic_cohort_with_rao_adapted_criteria.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv"

LITERAL_MIN_DAYS_BEFORE_DX = 7    # 1 week - Rao et al.'s actual published window
LITERAL_MAX_DAYS_BEFORE_DX = 90   # 3 months

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = [
    "first_K31_84_date", "first_GES_date",
    "has_any_typical_symptom_3to12mo_before_dx",
    "has_upper_endoscopy_1to12mo_before_dx",
    "ges_within_90d_before_dx",  # the already-computed modified (0-90) version
    "meets_rao_adapted_gp_criteria",  # the already-computed modified result
    "in_study_period",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing} - check this is the right input file (output of check_rao_adapted_gp_criteria.py).")

dx_date = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_date = pd.to_datetime(df["first_GES_date"], errors="coerce")
days_dx_after_ges = (dx_date - ges_date).dt.days

df["ges_within_7to90d_before_dx_LITERAL"] = (
    ges_date.notna() & dx_date.notna()
    & (days_dx_after_ges >= LITERAL_MIN_DAYS_BEFORE_DX)
    & (days_dx_after_ges <= LITERAL_MAX_DAYS_BEFORE_DX)
)

df["meets_rao_LITERAL_gp_criteria"] = (
    df["ges_within_7to90d_before_dx_LITERAL"]
    & df["has_any_typical_symptom_3to12mo_before_dx"]
    & df["has_upper_endoscopy_1to12mo_before_dx"]
)

in_period = df["in_study_period"]
n_total = in_period.sum()
print(f"\nOf the {n_total:,} in-study-period K31.84 patients:\n")

print("GES timing piece alone:")
print(f"  LITERAL (7-90 days before dx, as published):  {(in_period & df['ges_within_7to90d_before_dx_LITERAL']).sum():,}")
print(f"  MODIFIED (0-90 days before dx, same-day incl): {(in_period & df['ges_within_90d_before_dx']).sum():,}")

print("\nMeets ALL THREE criteria (GES timing + symptom + endoscopy):")
n_literal = (in_period & df['meets_rao_LITERAL_gp_criteria']).sum()
n_modified = (in_period & df['meets_rao_adapted_gp_criteria']).sum()
print(f"  LITERAL Rao et al. definition:  {n_literal:,} ({100 * n_literal / n_total:.1f}%)")
print(f"  MODIFIED (same-day included):   {n_modified:,} ({100 * n_modified / n_total:.1f}%)")
print(f"  difference: {n_modified - n_literal:,} patients gained by allowing same-day GES")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with both versions as separate columns.")
print(f"(Original {INPUT_CSV} left untouched.)")
