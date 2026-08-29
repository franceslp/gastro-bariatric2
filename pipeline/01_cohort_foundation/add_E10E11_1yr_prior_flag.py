"""
add_E10E11_1yr_prior_flag.py

Adds an exclusion flag column mirroring the K31.84 flag:
  exclude_E10E11_too_far_before_surgery: True = EXCLUDE (closest E10/E11
  more than 1 year before surgery); False = KEEP (within 1 year).

Same 1-year window as the K31.84 flag, same anchor logic.

Note: diabetes (E10/E11) is a chronic, rarely-resolving condition -
a long gap between diabetes diagnosis and surgery is less clinically
concerning than for gastroparesis. This flag is descriptive only and
should be discussed with Dr. Sujka before applying as a hard exclusion.

No new scan - pure arithmetic on columns already in the file.
"""

import pandas as pd

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_with_E10E11_1yr_flag.csv"

MAX_DAYS_BEFORE_SURGERY = 365

print(">>> SCRIPT VERSION: add_E10E11_1yr_prior_flag_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Cohort size: {len(df):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
e10e11_dt = pd.to_datetime(df["closest_E10_or_E11_strictly_before_surgery"], errors="coerce")

days_E10E11_before_surgery = (surgery_dt - e10e11_dt).dt.days
df["days_E10E11_before_surgery"] = days_E10E11_before_surgery
df["years_E10E11_before_surgery"] = (days_E10E11_before_surgery / 365.25).round(2)

# True = EXCLUDE (E10/E11 too far before surgery - more than 1 year)
# False = KEEP (E10/E11 within 1 year of surgery)
df["exclude_E10E11_too_far_before_surgery"] = (
    days_E10E11_before_surgery.isna() |
    (days_E10E11_before_surgery > MAX_DAYS_BEFORE_SURGERY)
)

n_keep = (~df["exclude_E10E11_too_far_before_surgery"]).sum()
n_exclude = df["exclude_E10E11_too_far_before_surgery"].sum()

n_after_surgery = (days_E10E11_before_surgery < 0).sum()
print(f"E10/E11 dated AFTER surgery (should be 0): {n_after_surgery:,}")

print(f"\nExclusion flag summary (E10/E11 must be within 1 year of surgery):")
print(f"  KEEP (closest E10/E11 within 1 year of surgery):        {n_keep:,}/{len(df):,} ({100*n_keep/len(df):.1f}%)")
print(f"  EXCLUDE (closest E10/E11 more than 1 year before surgery): {n_exclude:,}/{len(df):,} ({100*n_exclude/len(df):.1f}%)")

print("\nDays between closest E10/E11 and surgery - full distribution:")
print(df["days_E10E11_before_surgery"].describe())
print(f"\nMedian days E10/E11 before surgery: {df['days_E10E11_before_surgery'].median():,.0f}")
print(f"IQR: {df['days_E10E11_before_surgery'].quantile(0.25):,.0f} - {df['days_E10E11_before_surgery'].quantile(0.75):,.0f} days")
print(f"Median years E10/E11 before surgery: {df['years_E10E11_before_surgery'].median():.2f}")

print("\nPatients KEPT at each maximum-gap threshold:")
for threshold in [30, 90, 180, 365, 730]:
    n = (days_E10E11_before_surgery.notna() & (days_E10E11_before_surgery <= threshold)).sum()
    print(f"  <= {threshold} days: {n:,}/{len(df):,} kept ({100*n/len(df):.1f}%)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print("(Flag is descriptive only - no patients removed from this file.)")
