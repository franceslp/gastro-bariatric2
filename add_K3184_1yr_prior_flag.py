"""
add_K3184_1yr_prior_flag.py

Adds a descriptive column to the master cohort file:
  K3184_at_least_1yr_before_surgery: True if closest_K31_84_strictly_before_surgery
  is at least 365 days before bariatric_date, False otherwise.

Uses closest_K31_84_strictly_before_surgery as the anchor (conservative
choice - if even the most recent K31.84 before surgery clears the 1-year
threshold, we know a qualifying occurrence exists; if it doesn't, we flag
it for review).

No new scan - pure arithmetic on columns already in the file.
This is a DESCRIPTIVE flag only - not applied as a hard exclusion yet.
All exclusion criteria will be applied together in a final step.
"""

import pandas as pd

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_with_K3184_1yr_flag.csv"

MIN_DAYS_BEFORE_SURGERY = 365

print(">>> SCRIPT VERSION: add_K3184_1yr_prior_flag_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Cohort size: {len(df):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
k3184_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")

days_K3184_before_surgery = (surgery_dt - k3184_dt).dt.days
df["days_K3184_before_surgery"] = days_K3184_before_surgery
df["K3184_at_least_1yr_before_surgery"] = (
    days_K3184_before_surgery.notna() &
    (days_K3184_before_surgery >= MIN_DAYS_BEFORE_SURGERY)
)

n_meets = df["K3184_at_least_1yr_before_surgery"].sum()
n_fails = (~df["K3184_at_least_1yr_before_surgery"]).sum()

# Hard check: closest_K31_84_strictly_before_surgery was computed with
# date < surgery_date enforced during the scan - so negative values here
# should be impossible by construction. Verifying explicitly rather than
# assuming, same pattern used throughout this project today.
n_after_surgery = (days_K3184_before_surgery < 0).sum()
print(f"K31.84 dated AFTER surgery (should be 0 - would indicate upstream bug): {n_after_surgery:,}")

print(f"\nK31.84 at least 1 year ({MIN_DAYS_BEFORE_SURGERY} days) before surgery:")
print(f"  Meets threshold: {n_meets:,}/{len(df):,} ({100*n_meets/len(df):.1f}%)")
print(f"  Does NOT meet threshold: {n_fails:,}/{len(df):,} ({100*n_fails/len(df):.1f}%)")

print("\nDays between K31.84 and surgery - full distribution:")
print(df["days_K3184_before_surgery"].describe())
print(f"\nMedian days K31.84 before surgery: {df['days_K3184_before_surgery'].median():,.0f}")
print(f"IQR: {df['days_K3184_before_surgery'].quantile(0.25):,.0f} - {df['days_K3184_before_surgery'].quantile(0.75):,.0f} days")

# Years column - directly usable in manuscript tables without manual
# conversion: "Median time from K31.84 documentation to surgery was X years"
df["years_K3184_before_surgery"] = (days_K3184_before_surgery / 365.25).round(2)
print(f"Median years K31.84 before surgery: {df['years_K3184_before_surgery'].median():.2f}")

print("\nBreakdown by threshold:")
for threshold in [30, 90, 180, 365, 730]:
    n = (days_K3184_before_surgery.notna() & (days_K3184_before_surgery >= threshold)).sum()
    print(f"  >= {threshold} days: {n:,}/{len(df):,} ({100*n/len(df):.1f}%)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print("(Flag is descriptive only - no patients removed from this file.)")
