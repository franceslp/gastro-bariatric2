"""
check_multi_surgery_K3184_patients.py

Quick check: of the 2,842 bariatric-surgery patients, how many have K31.84
specifically in their chart (not just legacy 536.3) AND have multiple
distinct bariatric surgery dates? No new scan - uses columns already in
the master file.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)

if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

bariatric = df[df["has_bariatric_surgery"]].copy()
print(f"Total bariatric-surgery patients: {len(bariatric):,}")

has_k3184 = bariatric["first_K31_84_date"].notna()
print(f"  Of those, have K31.84 specifically (not just legacy 536.3): {has_k3184.sum():,}")
print(f"  Of those, 536.3-only (no K31.84 at all): {(~has_k3184).sum():,}")

multi_surgery = bariatric["num_distinct_bariatric_surgery_dates"] > 1
print(f"\nMultiple distinct surgery dates, among ALL bariatric-surgery patients: {multi_surgery.sum():,}")

k3184_and_multi_surgery = has_k3184 & multi_surgery
print(f"Multiple distinct surgery dates, AND have K31.84 in their chart:       {k3184_and_multi_surgery.sum():,}")

if k3184_and_multi_surgery.sum() > 0:
    print("\nThese patients (for reference):")
    cols = ["patient_id", "first_K31_84_date", "bariatric_date", "num_distinct_bariatric_surgery_dates", "bariatric_cpt_codes_seen"]
    print(bariatric[k3184_and_multi_surgery][cols].to_string(index=False))
