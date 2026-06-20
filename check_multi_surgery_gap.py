"""
check_multi_surgery_gap.py

Of the 110 multi-surgery patients, how many already qualify (their FIRST
surgery, which is what bariatric_date represents, already has a qualifying
K31.84 in the window) versus how many came back "no" - those are the ones
whose LATER surgery might actually qualify even though their first one
didn't, and need a follow-up check.

No new scan - just reading the already-completed window-check output.
"""

import pandas as pd

WINDOW_CHECK_CSV = "bariatric_patients_K3184_window_check.csv"

df = pd.read_csv(WINDOW_CHECK_CSV, dtype={"patient_id": str}, low_memory=False)

multi_surgery = df["num_distinct_bariatric_surgery_dates"] > 1
print(f"Multi-surgery patients: {multi_surgery.sum():,}")

multi_qualifying = multi_surgery & df["has_K3184_strictly_in_window"]
multi_not_qualifying = multi_surgery & ~df["has_K3184_strictly_in_window"]

print(f"  Already qualify using their FIRST surgery (no follow-up needed): {multi_qualifying.sum():,}")
print(f"  Do NOT qualify using their FIRST surgery (need follow-up vs. their LATER surgery): {multi_not_qualifying.sum():,}")

if multi_not_qualifying.sum() > 0:
    print("\nThese are the patients needing follow-up:")
    cols = ["patient_id", "first_K31_84_date", "bariatric_date", "num_distinct_bariatric_surgery_dates"]
    print(df[multi_not_qualifying][cols].to_string(index=False))
