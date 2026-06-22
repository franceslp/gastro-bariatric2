"""
build_funnel_csvs.py

Same funnel logic as build_funnel_excel.py but writes 6 separate CSVs
instead of one Excel file - no openpyxl needed.

Files written:
  funnel_1_all_patients_1118.csv
  funnel_2_after_age.csv
  funnel_3_after_K3184_timing.csv
  funnel_4_after_E10E11_timing.csv
  funnel_5_after_GES_1yr.csv
  funnel_6_final_cohort.csv
"""

import pandas as pd

MASTER_CSV = "master_cohort_FINAL_1118.csv"
EXCLUSION_CSV = "exclusion_criteria_sheet.csv"

print(">>> SCRIPT VERSION: build_funnel_csvs_v1 <<<")

master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Master file: {len(master):,} patients, {len(master.columns)} columns")

excl = pd.read_csv(EXCLUSION_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id",
                             "exclude_age_under_18",
                             "exclude_K3184_too_far_before_surgery",
                             "exclude_E10E11_too_far_before_surgery",
                             "exclude_no_GES_within_1yr",
                             "exclude_no_prokinetic_within_1yr",
                             "multi_surgery_flag",
                             "all_bariatric_surgery_dates",
                             "second_bariatric_surgery_date"])

BOOL_MAP = {True: True, False: False, "True": True, "False": False}
bool_cols = ["exclude_age_under_18", "exclude_K3184_too_far_before_surgery",
             "exclude_E10E11_too_far_before_surgery",
             "exclude_no_GES_within_1yr", "exclude_no_prokinetic_within_1yr"]
for c in bool_cols:
    excl[c] = excl[c].map(BOOL_MAP).fillna(False).astype(bool)

df = master.merge(excl, on="patient_id", how="left")
assert len(df) == 1118, f"Row count changed: {len(df)}"

sheet1 = df.copy()
sheet2 = sheet1[~sheet1["exclude_age_under_18"]].copy()
sheet3 = sheet2[~sheet2["exclude_K3184_too_far_before_surgery"]].copy()
sheet4 = sheet3[~sheet3["exclude_E10E11_too_far_before_surgery"]].copy()
sheet5 = sheet4[~sheet4["exclude_no_GES_within_1yr"]].copy()
sheet6 = sheet5[~sheet5["exclude_no_prokinetic_within_1yr"]].copy()

print(f"\nCONSORT FUNNEL:")
print(f"  1 - All patients:                    {len(sheet1):,}")
print(f"  2 - After age exclusion:             {len(sheet2):,} (-{len(sheet1)-len(sheet2):,})")
print(f"  3 - After K31.84 timing:             {len(sheet3):,} (-{len(sheet2)-len(sheet3):,})")
print(f"  4 - After E10/E11 timing:            {len(sheet4):,} (-{len(sheet3)-len(sheet4):,})")
print(f"  5 - After GES within 1yr:            {len(sheet5):,} (-{len(sheet4)-len(sheet5):,})")
print(f"  6 - Final cohort (after prokinetic): {len(sheet6):,} (-{len(sheet5)-len(sheet6):,})")

files = [
    ("funnel_1_all_patients_1118.csv", sheet1),
    ("funnel_2_after_age.csv", sheet2),
    ("funnel_3_after_K3184_timing.csv", sheet3),
    ("funnel_4_after_E10E11_timing.csv", sheet4),
    ("funnel_5_after_GES_1yr.csv", sheet5),
    ("funnel_6_final_cohort.csv", sheet6),
]

print("\nWriting CSVs...")
for filename, data in files:
    data.to_csv(filename, index=False)
    print(f"  Wrote {filename} ({len(data):,} rows)")

print("\nDone. Copy all 6 files to GCS:")
print("gsutil cp funnel_*.csv gs://test-skynet-lh/frances-perez/results/")
