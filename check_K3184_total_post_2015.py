"""
check_K3184_total_post_2015.py

Of the 289,647 K31.84-ever patients (full 2,134,876-patient population, no
bariatric surgery restriction), how many have ANY K31.84 occurrence on/after
Oct 1, 2015 - using the CORRECTED any-occurrence logic (last_K31_84_date),
not the old flawed first-occurrence-only count (263,087).
"""

import pandas as pd

STUDY_START = pd.Timestamp("2015-10-01")
BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "first_K31_84_date", "last_K31_84_date", "in_study_period"])

has_k3184 = df["first_K31_84_date"].notna()
print(f"K31.84-ever patients (any time, any date): {has_k3184.sum():,}")

last_dt = pd.to_datetime(df["last_K31_84_date"], errors="coerce")
any_occurrence_post_2015 = has_k3184 & last_dt.notna() & (last_dt >= STUDY_START)
print(f"  Of those, with ANY occurrence on/after Oct 1 2015 (corrected logic): {any_occurrence_post_2015.sum():,}")

# For reference, the old flawed first-occurrence-only count
old_flawed = has_k3184 & df["in_study_period"]
print(f"  (For reference, old flawed first-occurrence-only count: {old_flawed.sum():,})")
print(f"  Difference (patients recovered by the fix): {any_occurrence_post_2015.sum() - old_flawed.sum():,}")
