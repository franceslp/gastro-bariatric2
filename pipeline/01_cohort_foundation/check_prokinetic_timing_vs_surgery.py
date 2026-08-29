"""
check_prokinetic_timing_vs_surgery.py

The prokinetic scan found drugs within 1 year AFTER the K31.84 anchor
(closest_K31_84_strictly_before_surgery). But since that anchor could be
anywhere from 1 day to 365 days before surgery, a prokinetic that starts
within 1 year of the K31.84 anchor could actually be AFTER the surgery
date itself.

This checks:
  1. How many qualifying prokinetics fall before surgery (expected majority)
  2. How many fall after surgery (possible, worth knowing)
  3. How many fall on the same day as surgery
  4. Days from prokinetic start to surgery date (positive = before surgery,
     negative = after surgery)

No new scan - pure arithmetic on existing columns.
"""

import pandas as pd

PROKINETIC_CSV = "cohort_prokinetic_after_closest_dx_1yr.csv"
COHORT_CSV = "final_cohort_with_age.csv"
OUTPUT_CSV = "prokinetic_timing_vs_surgery_check.csv"

print(">>> SCRIPT VERSION: check_prokinetic_timing_vs_surgery_v1 <<<")

prok = pd.read_csv(PROKINETIC_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "prokinetic_after_closest_dx_1yr_date",
                             "prokinetic_after_closest_dx_1yr_drug",
                             "days_to_prokinetic_after_K3184",
                             "closest_K31_84_strictly_before_surgery"])

cohort = pd.read_csv(COHORT_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "bariatric_date"])

df = prok.merge(cohort, on="patient_id", how="left")
print(f"Cohort size: {len(df):,} patients")
print(f"Missing bariatric date after merge (should be 0): {df['bariatric_date'].isna().sum():,}")

prok_dt = pd.to_datetime(df["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
k3184_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")

has_prokinetic = prok_dt.notna()
n_with_prokinetic = has_prokinetic.sum()
print(f"Patients with a qualifying prokinetic: {n_with_prokinetic:,}/{len(df):,}")

days_prok_to_surgery = (surgery_dt - prok_dt).dt.days
df["days_prokinetic_before_surgery"] = days_prok_to_surgery

n_before_surgery = (has_prokinetic & (days_prok_to_surgery > 0)).sum()
n_same_day_surgery = (has_prokinetic & (days_prok_to_surgery == 0)).sum()
n_after_surgery = (has_prokinetic & (days_prok_to_surgery < 0)).sum()

print(f"\nOf the {n_with_prokinetic:,} patients with a qualifying prokinetic:")
print(f"  Prokinetic BEFORE surgery:      {n_before_surgery:,} ({100*n_before_surgery/n_with_prokinetic:.1f}%)")
print(f"  Prokinetic SAME DAY as surgery: {n_same_day_surgery:,} ({100*n_same_day_surgery/n_with_prokinetic:.1f}%)")
print(f"  Prokinetic AFTER surgery:       {n_after_surgery:,} ({100*n_after_surgery/n_with_prokinetic:.1f}%)")

# More meaningful for methods: pre-op prokinetic exposure as a fraction of
# the FULL cohort, not just those who had any qualifying prokinetic at all.
n_preop = n_before_surgery + n_same_day_surgery
print(f"\nPre-op prokinetic exposure (before or same day as surgery): "
      f"{n_preop:,}/{len(df):,} ({100*n_preop/len(df):.1f}% of full cohort)")

if n_after_surgery > 0:
    after_df = df[has_prokinetic & (days_prok_to_surgery < 0)][
        ["patient_id", "closest_K31_84_strictly_before_surgery",
         "prokinetic_after_closest_dx_1yr_date", "bariatric_date",
         "days_to_prokinetic_after_K3184", "days_prokinetic_before_surgery"]
    ]
    print(f"\nPatients with prokinetic AFTER surgery:")
    print(after_df.to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
