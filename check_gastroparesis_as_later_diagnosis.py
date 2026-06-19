"""
check_gastroparesis_as_later_diagnosis.py

Of the patients who meet the 5yr E10/E11 concurrency rule, how many had
GASTROPARESIS (K31.84) as the LATER of the two diagnoses (vs diabetes being
the later one)? And for those patients specifically, how many days before
surgery was their gastroparesis diagnosis?

No new scan - everything needed is already in bariatric_subset_5yr_concurrency_E10_E11.csv.
"""

import pandas as pd

INPUT_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

# Restrict to in-study-period K31.84 patients (diagnosed on/after Oct 2015) -
# matches the 1,126 figure from yesterday's run. Without this filter, the
# count includes patients whose K31.84 date falls before the study period,
# which yesterday's reported number excluded.
qualified = df[(df["meets_5yr_rule_E10_E11"] == True) & (df["in_study_period"] == True)].copy()
print(f"Patients meeting the 5yr E10/E11 rule (in-study-period only): {len(qualified):,}\n")

k3184_dt = pd.to_datetime(qualified["first_K31_84_date"], errors="coerce")
diabetes_dt = pd.to_datetime(qualified["first_E10_or_E11_date"], errors="coerce")
surgery_dt = pd.to_datetime(qualified["bariatric_date"], errors="coerce")

# Which one was later (= the "concurrent date" that the 5yr rule measured against surgery)
qualified["gastroparesis_is_later"] = k3184_dt >= diabetes_dt
qualified["diabetes_is_later"] = diabetes_dt > k3184_dt

n_gp_later = qualified["gastroparesis_is_later"].sum()
n_diabetes_later = qualified["diabetes_is_later"].sum()
print(f"Gastroparesis (K31.84) was the LATER diagnosis: {n_gp_later:,} patients")
print(f"Diabetes (E10/E11) was the LATER diagnosis:     {n_diabetes_later:,} patients")

# For the gastroparesis-later group, how far out from surgery was that diagnosis?
gp_later_group = qualified[qualified["gastroparesis_is_later"]].copy()
days_before_surgery = gp_later_group["days_concurrent_E10_E11_to_surgery"]

print(f"\nFor the {len(gp_later_group):,} patients where gastroparesis was the later diagnosis,")
print("days between gastroparesis diagnosis and surgery:")
print(f"  mean:   {days_before_surgery.mean():,.0f} days (~{days_before_surgery.mean()/365.25:.1f} years)")
print(f"  median: {days_before_surgery.median():,.0f} days (~{days_before_surgery.median()/365.25:.1f} years)")
print(f"  min:    {days_before_surgery.min():,.0f} days")
print(f"  max:    {days_before_surgery.max():,.0f} days (~{days_before_surgery.max()/365.25:.1f} years)")

print("\nBreakdown by year before surgery:")
bins = [0, 365, 730, 1095, 1460, 1826]
labels = ["0-1 yr", "1-2 yr", "2-3 yr", "3-4 yr", "4-5 yr"]
bucketed = pd.cut(days_before_surgery, bins=bins, labels=labels, include_lowest=True)
print(bucketed.value_counts().sort_index())
