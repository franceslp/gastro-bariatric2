"""
add_strict_concurrency_sensitivity_check.py

Adds a "strict" (interpretation B) sensitivity-analysis version of the
diabetes-gastroparesis concurrency flags already in the cohort file.

Interpretation A (already in the file, meets_1yr/5yr_concurrency_rule):
    the LATER of (first_diabetes_dx_date, first_K31_84_date) must fall
    within the window before bariatric_date.

Interpretation B (this script, the "_strict" suffix):
    EACH of first_diabetes_dx_date AND first_K31_84_date must
    INDIVIDUALLY fall within the window before bariatric_date - a
    stricter requirement, since a patient with a much older diagnosis of
    one condition (even if the other is recent) would no longer qualify.

This is pure arithmetic on columns already in the cohort file -
first_diabetes_dx_date, first_K31_84_date, and bariatric_date are all
already there. No GCS access needed, runs in seconds. Writes a NEW file
rather than overwriting the input.

Run this AFTER add_bariatric_surgery_and_sadda_concurrency.py.
"""

import os
import pandas as pd

INPUT_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_FULL_with_concurrency_sensitivity.csv"

CONCURRENCY_WINDOWS_YEARS = {"1yr": 1, "5yr": 5}
CONCURRENCY_WINDOWS_DAYS = {
    label: round(years * 365.25) for label, years in CONCURRENCY_WINDOWS_YEARS.items()
}  # {"1yr": 365, "5yr": 1826}

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Missing required input file: {INPUT_CSV}")

print(f"Loading {INPUT_CSV}...")
cohort_df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = ["first_K31_84_date", "first_diabetes_dx_date", "bariatric_date", "has_bariatric_surgery"]
missing_cols = [c for c in required_cols if c not in cohort_df.columns]
if missing_cols:
    raise ValueError(f"{INPUT_CSV} is missing expected columns: {missing_cols}")

first_k3184_dt = pd.to_datetime(cohort_df["first_K31_84_date"], errors="coerce")
first_diabetes_dt = pd.to_datetime(cohort_df["first_diabetes_dx_date"], errors="coerce")
bariatric_dt = pd.to_datetime(cohort_df["bariatric_date"], errors="coerce")

days_diabetes_to_surgery = (bariatric_dt - first_diabetes_dt).dt.days
days_k3184_to_surgery = (bariatric_dt - first_k3184_dt).dt.days

cohort_df["days_diabetes_dx_to_bariatric_surgery"] = days_diabetes_to_surgery
cohort_df["days_K31_84_to_bariatric_surgery"] = days_k3184_to_surgery

all_dates_present = (
    cohort_df["has_bariatric_surgery"]
    & first_k3184_dt.notna()
    & first_diabetes_dt.notna()
    & bariatric_dt.notna()
)

for label, window_days in CONCURRENCY_WINDOWS_DAYS.items():
    diabetes_in_window = (days_diabetes_to_surgery >= 0) & (days_diabetes_to_surgery <= window_days)
    k3184_in_window = (days_k3184_to_surgery >= 0) & (days_k3184_to_surgery <= window_days)
    cohort_df[f"meets_{label}_concurrency_rule_strict"] = (
        all_dates_present & diabetes_in_window & k3184_in_window
    )

print("\nSANITY CHECK (interpretation A vs B side by side):")
for label in CONCURRENCY_WINDOWS_DAYS:
    n_strict = cohort_df[f"meets_{label}_concurrency_rule_strict"].sum()
    col_a = f"meets_{label}_concurrency_rule"
    if col_a in cohort_df.columns:
        n_a = cohort_df[col_a].sum()
        print(f"  {label}: interpretation A (later-date only) = {n_a:,}   |   interpretation B (both individually) = {n_strict:,}")
    else:
        print(f"  {label}: interpretation B (both individually) = {n_strict:,}  (interpretation A column not found in input)")

cohort_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with days_diabetes_dx_to_bariatric_surgery,")
print("days_K31_84_to_bariatric_surgery, meets_1yr_concurrency_rule_strict, and")
print("meets_5yr_concurrency_rule_strict columns added.")
print(f"(Original {INPUT_CSV} left untouched - use the new file as input to the next step.)")
