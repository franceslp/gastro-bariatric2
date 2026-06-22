"""
build_funnel_excel.py

Builds a multi-sheet Excel workbook showing the CONSORT-style funnel,
with each sheet representing the population after each successive
exclusion criterion is applied.

Sheet 1: All Patients (1,118)
Sheet 2: After Age Exclusion
Sheet 3: After K31.84 Timing (<=1yr before surgery)
Sheet 4: After E10/E11 Timing (<=1yr before surgery)
Sheet 5: After GES (within 1yr of K31.84)
Sheet 6: Final Cohort (after prokinetic within 1yr of K31.84)

Each sheet is CUMULATIVE - showing who remains after all criteria
applied so far, not just who fails that one criterion in isolation.
"""

import pandas as pd

MASTER_CSV = "master_cohort_FINAL_1118.csv"
EXCLUSION_CSV = "exclusion_criteria_sheet.csv"
OUTPUT_XLSX = "cohort_funnel_FINAL.xlsx"

print(">>> SCRIPT VERSION: build_funnel_excel_v1 <<<")

# Load master file (all 25 columns for every sheet)
master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Master file: {len(master):,} patients, {len(master.columns)} columns")

# Load exclusion flags
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

# Merge exclusion flags onto master
df = master.merge(excl, on="patient_id", how="left")
assert len(df) == 1118, f"Row count changed after merge: {len(df)}"

# Build each cumulative sheet
sheet1 = df.copy()

sheet2 = sheet1[~sheet1["exclude_age_under_18"]].copy()

sheet3 = sheet2[~sheet2["exclude_K3184_too_far_before_surgery"]].copy()

sheet4 = sheet3[~sheet3["exclude_E10E11_too_far_before_surgery"]].copy()

sheet5 = sheet4[~sheet4["exclude_no_GES_within_1yr"]].copy()

sheet6 = sheet5[~sheet5["exclude_no_prokinetic_within_1yr"]].copy()

# Print funnel summary
print(f"\nCONSORT FUNNEL:")
print(f"  Sheet 1 - All patients:                    {len(sheet1):,}")
print(f"  Sheet 2 - After age exclusion:             {len(sheet2):,} (-{len(sheet1)-len(sheet2):,})")
print(f"  Sheet 3 - After K31.84 timing:             {len(sheet3):,} (-{len(sheet2)-len(sheet3):,})")
print(f"  Sheet 4 - After E10/E11 timing:            {len(sheet4):,} (-{len(sheet3)-len(sheet4):,})")
print(f"  Sheet 5 - After GES within 1yr:            {len(sheet5):,} (-{len(sheet4)-len(sheet5):,})")
print(f"  Sheet 6 - Final cohort (after prokinetic): {len(sheet6):,} (-{len(sheet5)-len(sheet6):,})")

# Multi-surgery check in final cohort
n_multi_final = sheet6["multi_surgery_flag"].sum() if "multi_surgery_flag" in sheet6.columns else 0
print(f"\n  Multi-surgery patients in final cohort: {n_multi_final:,}")

# Write Excel workbook
print(f"\nWriting {OUTPUT_XLSX}...")
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    sheet1.to_excel(writer, sheet_name="1_All_Patients_1118", index=False)
    sheet2.to_excel(writer, sheet_name="2_After_Age", index=False)
    sheet3.to_excel(writer, sheet_name="3_After_K3184_Timing", index=False)
    sheet4.to_excel(writer, sheet_name="4_After_E10E11_Timing", index=False)
    sheet5.to_excel(writer, sheet_name="5_After_GES_1yr", index=False)
    sheet6.to_excel(writer, sheet_name="6_Final_Cohort", index=False)

print(f"Wrote {OUTPUT_XLSX}")
print("Done - 6 sheets, each showing cumulative exclusions applied.")
