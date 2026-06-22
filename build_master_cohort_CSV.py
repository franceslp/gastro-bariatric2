"""
build_master_cohort_CSV.py

Merges all computed columns into one clean master file for the 1,118-patient
cohort, with columns in the exact order specified. No new scans - pure merge
of files already built and verified.

Sources:
  - final_cohort_with_age.csv (base - age, surgery date)
  - cohort_with_demographics.csv (sex, race, ethnicity, marital, deceased)
  - bariatric_patients_K3184_window_check.csv (bariatric CPT codes)
  - cohort_closest_K3184_and_diabetes_to_surgery.csv (closest K31.84, diabetes type)
  - cohort_K3184_gap_analysis.csv (num encounters, span, all dates)
  - cohort_with_E10_E11_specific_codes.csv (closest E10/E11 code + date)
  - cohort_GES_before_closest_dx_1yr.csv (GES code + date)
  - cohort_prokinetic_with_surgery_timing_flags.csv (prokinetic drug, date, timing flags)
"""

import pandas as pd

OUTPUT_CSV = "master_cohort_FINAL_1118.csv"
EXPECTED_ROWS = 1118

print(">>> SCRIPT VERSION: build_master_cohort_CSV_v1 <<<")

print("Loading source files...")

base = pd.read_csv("final_cohort_with_age.csv", dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "year_of_birth", "age_at_surgery_approx",
                             "bariatric_date", "meets_age_requirement"])

demo = pd.read_csv("cohort_with_demographics.csv", dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "sex", "race", "ethnicity",
                             "marital_status", "month_year_death", "deceased"])

bariatric = pd.read_csv("bariatric_patients_K3184_window_check.csv", dtype={"patient_id": str},
                         low_memory=False, usecols=["patient_id", "bariatric_cpt_codes_seen"])

closest_dx = pd.read_csv("cohort_closest_K3184_and_diabetes_to_surgery.csv",
                          dtype={"patient_id": str}, low_memory=False,
                          usecols=["patient_id", "closest_K31_84_strictly_before_surgery",
                                   "diabetes_type_label"])

gap = pd.read_csv("cohort_K3184_gap_analysis.csv", dtype={"patient_id": str}, low_memory=False,
                   usecols=["patient_id", "num_K31_84_encounters", "K3184_span_days",
                             "all_K31_84_dates"])

e10e11 = pd.read_csv("cohort_with_E10_E11_specific_codes.csv", dtype={"patient_id": str},
                      low_memory=False,
                      usecols=["patient_id", "closest_E10_E11_code_before_surgery",
                               "closest_E10_E11_date_before_surgery_v2"])

ges = pd.read_csv("cohort_GES_before_K3184_no_limit.csv", dtype={"patient_id": str},
                   low_memory=False,
                   usecols=["patient_id", "closest_GES_before_K3184_dx_code",
                             "closest_GES_before_K3184_dx_date",
                             "days_GES_before_K3184"])

prok = pd.read_csv("cohort_prokinetic_after_K3184_no_limit.csv",
                    dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "first_prokinetic_after_K3184_dx_drug",
                             "first_prokinetic_after_K3184_dx_date",
                             "days_to_prokinetic_after_K3184"])

print("  all files loaded")

# --- Merge all into base ---
print("\nMerging...")
df = base.copy()
for name, src in [("demographics", demo), ("bariatric codes", bariatric),
                   ("closest K31.84/diabetes", closest_dx), ("gap analysis", gap),
                   ("E10/E11 codes", e10e11), ("GES", ges), ("prokinetic", prok)]:
    overlap = set(df.columns).intersection(src.columns) - {"patient_id"}
    if overlap:
        raise ValueError(f"Column collision merging {name}: {overlap}")
    df = df.merge(src, on="patient_id", how="left")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Row count changed to {len(df):,} after merging {name} - investigate.")

print(f"  merged successfully, {len(df):,} rows confirmed throughout")

# --- Rename columns to final clean names ---
df = df.rename(columns={
    "closest_E10_E11_date_before_surgery_v2": "closest_E10_E11_date_before_surgery",
})

# --- Reorder columns exactly as specified ---
COLUMN_ORDER = [
    "patient_id",
    "year_of_birth",
    "age_at_surgery_approx",
    "bariatric_date",
    "bariatric_cpt_codes_seen",
    "closest_K31_84_strictly_before_surgery",
    "num_K31_84_encounters",
    "K3184_span_days",
    "all_K31_84_dates",
    "diabetes_type_label",
    "closest_E10_E11_code_before_surgery",
    "closest_E10_E11_date_before_surgery",
    "closest_GES_before_K3184_dx_code",
    "closest_GES_before_K3184_dx_date",
    "days_GES_before_K3184",
    "first_prokinetic_after_K3184_dx_drug",
    "first_prokinetic_after_K3184_dx_date",
    "days_to_prokinetic_after_K3184",
    "sex",
    "race",
    "ethnicity",
    "marital_status",
    "month_year_death",
    "deceased",
    "meets_age_requirement",
]

remaining = [c for c in df.columns if c not in COLUMN_ORDER]
if remaining:
    print(f"\nExtra columns not in specified order (appended at end): {remaining}")

df = df[COLUMN_ORDER + remaining]

# --- Final QA ---
print(f"\nFinal QA:")
print(f"  Rows: {len(df):,} (expected {EXPECTED_ROWS:,})")
print(f"  Columns: {len(df.columns)}")
print(f"  All specified columns present: {all(c in df.columns for c in COLUMN_ORDER)}")
print(f"\nColumn order:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:2}. {c}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
