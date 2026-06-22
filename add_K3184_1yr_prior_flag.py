"""
add_K3184_1yr_prior_flag.py

Adds an exclusion flag column to the master cohort file:
  exclude_K3184_too_far_before_surgery: True = EXCLUDE (closest K31.84
  more than 1 year before surgery - too far from the surgical decision);
  False = KEEP (closest K31.84 within 1 year of surgery - actively
  documented close to the surgical decision, which is what we want).

No new scan - pure arithmetic on columns already in the file.
This is a DESCRIPTIVE flag only - not applied as a hard exclusion yet.
All exclusion criteria will be applied together in a final step.
"""

import pandas as pd

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_with_K3184_1yr_flag.csv"

MAX_DAYS_BEFORE_SURGERY = 365  # K31.84 more than this many days before surgery = exclude

print(">>> SCRIPT VERSION: add_K3184_1yr_prior_flag_v2 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Cohort size: {len(df):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
k3184_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")

days_K3184_before_surgery = (surgery_dt - k3184_dt).dt.days
df["days_K3184_before_surgery"] = days_K3184_before_surgery
df["years_K3184_before_surgery"] = (days_K3184_before_surgery / 365.25).round(2)

# True = EXCLUDE (K31.84 too far before surgery - more than 1 year)
# False = KEEP (K31.84 within 1 year of surgery, actively documented
#         close to the surgical decision - which is what we want)
df["exclude_K3184_too_far_before_surgery"] = (
    days_K3184_before_surgery.isna() |
    (days_K3184_before_surgery > MAX_DAYS_BEFORE_SURGERY)
)

n_keep = (~df["exclude_K3184_too_far_before_surgery"]).sum()
n_exclude = df["exclude_K3184_too_far_before_surgery"].sum()

# Hard check: closest_K31_84_strictly_before_surgery was computed with
# date < surgery_date enforced during the scan - so negative values here
# should be impossible by construction. Verifying explicitly rather than
# assuming, same pattern used throughout this project today.
n_after_surgery = (days_K3184_before_surgery < 0).sum()
print(f"K31.84 dated AFTER surgery (should be 0 - would indicate upstream bug): {n_after_surgery:,}")

print(f"\nExclusion flag summary (K31.84 must be within 1 year of surgery):")
print(f"  KEEP (closest K31.84 within 1 year of surgery):        {n_keep:,}/{len(df):,} ({100*n_keep/len(df):.1f}%)")
print(f"  EXCLUDE (closest K31.84 more than 1 year before surgery): {n_exclude:,}/{len(df):,} ({100*n_exclude/len(df):.1f}%)")

print("\nDays between K31.84 and surgery - full distribution:")
print(df["days_K3184_before_surgery"].describe())
print(f"\nMedian days K31.84 before surgery: {df['days_K3184_before_surgery'].median():,.0f}")
print(f"IQR: {df['days_K3184_before_surgery'].quantile(0.25):,.0f} - {df['days_K3184_before_surgery'].quantile(0.75):,.0f} days")
print(f"Median years K31.84 before surgery: {df['years_K3184_before_surgery'].median():.2f}")

print("\nPatients KEPT at each maximum-gap threshold (closer = more kept):")
for threshold in [30, 90, 180, 365, 730]:
    n = (days_K3184_before_surgery.notna() & (days_K3184_before_surgery <= threshold)).sum()
    print(f"  <= {threshold} days: {n:,}/{len(df):,} kept ({100*n/len(df):.1f}%)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print("(Flag is descriptive only - no patients removed from this file.)")
