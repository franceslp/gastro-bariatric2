"""
check_110_multi_surgery.py

Of the 110 patients passing all exclusion criteria, how many are
multi-surgery patients? Flagged for Dr. Sujka review.
"""

import pandas as pd

EXCLUSION_CSV = "exclusion_criteria_sheet.csv"
MASTER_CSV = "master_cohort_FINAL_1118.csv"
BARIATRIC_CSV = "bariatric_patients_K3184_window_check.csv"

print(">>> SCRIPT VERSION: check_110_multi_surgery_v1 <<<")

excl = pd.read_csv(EXCLUSION_CSV, dtype={"patient_id": str}, low_memory=False)
master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "bariatric_cpt_codes_seen", "bariatric_date"])
bariatric = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "num_distinct_bariatric_surgery_dates"])

bool_cols = ["exclude_age_under_18", "exclude_K3184_too_far_before_surgery",
             "exclude_E10E11_too_far_before_surgery",
             "exclude_no_GES_within_1yr", "exclude_no_prokinetic_within_1yr"]
BOOL_MAP = {True: True, False: False, "True": True, "False": False}
for c in bool_cols:
    excl[c] = excl[c].map(BOOL_MAP).fillna(False).astype(bool)

passes_all = ~(excl[bool_cols].any(axis=1))
passing_ids = set(excl.loc[passes_all, "patient_id"])
print(f"Patients passing all exclusion criteria: {len(passing_ids):,}")

df = pd.DataFrame({"patient_id": list(passing_ids)})
df = df.merge(bariatric, on="patient_id", how="left")
df = df.merge(master, on="patient_id", how="left")

multi = df["num_distinct_bariatric_surgery_dates"] > 1
n_multi = multi.sum()
print(f"Of those, multi-surgery patients: {n_multi:,}/{len(df):,}")

if n_multi > 0:
    print("\nMulti-surgery patients in the passing cohort:")
    print(df[multi][["patient_id", "bariatric_date", "bariatric_cpt_codes_seen",
                      "num_distinct_bariatric_surgery_dates"]].to_string(index=False))
else:
    print("No multi-surgery patients in the passing cohort.")
