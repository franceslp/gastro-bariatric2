"""
build_master_cohort_file.py

PRIMARY COHORT DEFINITION (as clarified):
  1. K31.84 gastroparesis, in-study-period (on/after 2015-10-01)
  2. Bariatric surgery AFTER the K31.84 diagnosis date
  3. E10 or E11 (Type 1/Type 2) diabetes diagnosed BEFORE the surgery date

E08-E13 (broad diabetes) is NOT a requirement for inclusion - it's attached
as an informational flag only, so you can see how many patients would
additionally/alternately qualify under the broader diabetes definition.

ALSO ATTACHED as columns (not requirements, just visible for every patient
in the primary cohort): Definition 1 (prokinetic+GES), Rao literal, Rao
modified, the 5yr E10/E11 concurrency rule, the existing 1yr/5yr E08-E13
concurrency rules. This way the file is filterable to any of the stricter
definitions later without rebuilding anything.

No new GCS scan - this merges three files you already have:
  - gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv
  - bariatric_subset_5yr_concurrency_E10_E11.csv
  - gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv
"""

import pandas as pd
from pandas.api.types import is_bool_dtype

print(">>> SCRIPT VERSION: build_master_cohort_file_v1 <<<")

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
E10E11_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"
RAO_CSV = "gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv"
OUTPUT_CSV = "master_cohort_K3184_diabetes_before_surgery.csv"

# --- Load and merge ---
df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)

e10e11_df = pd.read_csv(E10E11_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "first_E10_date", "first_E11_date",
                                  "first_E10_or_E11_date", "meets_5yr_rule_E10_E11"])

rao_df = pd.read_csv(RAO_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "meets_rao_LITERAL_gp_criteria", "meets_rao_adapted_gp_criteria"])

df = df.merge(e10e11_df, on="patient_id", how="left")
df = df.merge(rao_df, on="patient_id", how="left")

n_unique = df["patient_id"].nunique()
if n_unique != len(df):
    raise ValueError(f"Merge produced {len(df):,} rows but only {n_unique:,} unique patient_ids - duplicate rows in an input file.")
print(f"Merged: {len(df):,} patients, no duplication from the merge")

if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")
if not is_bool_dtype(df["in_study_period"]):
    df["in_study_period"] = df["in_study_period"].astype(str).str.strip().str.lower().eq("true")

# QA: did every patient in the bariatric-surgery population actually get an
# E10/E11 and Rao record attached? (has_bariatric_surgery patients should
# all be present in e10e11_df, since that file was built from this exact
# subset; if any are missing here, the left-merge silently dropped data.)
for col in ["first_E10_date", "first_E11_date", "meets_rao_LITERAL_gp_criteria", "meets_rao_adapted_gp_criteria"]:
    n_missing = (df["has_bariatric_surgery"] & df[col].isna()).sum()
    print(f"  Missing {col} among bariatric-surgery patients: {n_missing:,}")
print()

# --- Dates ---
dx_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
ges_dt = pd.to_datetime(df["first_GES_date"], errors="coerce")
e08e13_dt = pd.to_datetime(df["first_diabetes_dx_date"], errors="coerce")
e10_dt = pd.to_datetime(df["first_E10_date"], errors="coerce")
e11_dt = pd.to_datetime(df["first_E11_date"], errors="coerce")

# --- PRIMARY COHORT CRITERIA ---
surgery_after_dx = dx_dt.notna() & surgery_dt.notna() & (surgery_dt > dx_dt)
df["surgery_after_K31_84_dx"] = surgery_after_dx

e10_before_surgery = e10_dt.notna() & surgery_dt.notna() & (e10_dt < surgery_dt)
e11_before_surgery = e11_dt.notna() & surgery_dt.notna() & (e11_dt < surgery_dt)
E10_E11_diabetes_before_surgery = e10_before_surgery | e11_before_surgery
df["E10_E11_diabetes_before_surgery"] = E10_E11_diabetes_before_surgery

primary_cohort = (
    df["in_study_period"]
    & df["has_bariatric_surgery"]
    & surgery_after_dx
    & E10_E11_diabetes_before_surgery
)
df["in_primary_cohort"] = primary_cohort

n_primary = primary_cohort.sum()
print("="*70)
print(f"PRIMARY COHORT (K31.84 + surgery after dx + E10/E11 diabetes before surgery): {n_primary:,}")
print("="*70)

# --- E08-E13 informational flag (NOT a requirement, just visible) ---
e08e13_before_surgery = e08e13_dt.notna() & surgery_dt.notna() & (e08e13_dt < surgery_dt)
df["E08_E13_diabetes_before_surgery"] = e08e13_before_surgery
n_e08e13_also = (primary_cohort & e08e13_before_surgery).sum()
n_e08e13_not = (primary_cohort & ~e08e13_before_surgery).sum()
print(f"\nOf the primary cohort, {n_e08e13_also:,}/{n_primary:,} also have E08-E13 (broad) diabetes before surgery")
print(f"  ({n_e08e13_not:,} do not - checked again below, right before saving, as a hard validation)")

# --- Other flags, attached for every patient (informational, not gating) ---
ges_before_or_same_day = ges_dt.notna() & dx_dt.notna() & (ges_dt <= dx_dt)
df["def1_met"] = (
    df["first_K31_84_date"].notna()
    & df["any_prokinetic_ever_after_dx"].fillna(False)
    & ges_before_or_same_day
)

print("\nFlag summary within the primary cohort:")
for label, col in [
    ("Definition 1 (prokinetic + GES)", "def1_met"),
    ("Rao, literal", "meets_rao_LITERAL_gp_criteria"),
    ("Rao, modified", "meets_rao_adapted_gp_criteria"),
    ("5yr E10/E11 concurrency rule", "meets_5yr_rule_E10_E11"),
    ("1yr E08-E13 concurrency rule", "meets_1yr_concurrency_rule"),
    ("5yr E08-E13 concurrency rule", "meets_5yr_concurrency_rule"),
]:
    if col in df.columns:
        n = (primary_cohort & df[col].fillna(False)).sum()
        print(f"  {label}: {n:,}/{n_primary:,}")

# Conceptual transparency: this cohort requires diabetes BEFORE SURGERY, but
# does NOT require diabetes before the gastroparesis diagnosis specifically.
# A patient with GP in 2017, diabetes in 2018, surgery in 2019 passes this
# cohort's criteria. That may be intentional, but it should be visible, not
# silent - so the actual ordering is captured as its own column rather than
# assumed. (Same pattern as the diagnosis-order columns built earlier today
# for the other E10/E11 population.)
e10e11_dt = pd.to_datetime(df["first_E10_or_E11_date"], errors="coerce")
both_dates_present = e10e11_dt.notna() & dx_dt.notna()
diagnosis_order = pd.Series("missing_one_or_both", index=df.index, dtype=object)
days_gp_minus_diabetes = (dx_dt - e10e11_dt).dt.days
diagnosis_order[both_dates_present & (days_gp_minus_diabetes < 0)] = "gastroparesis_first"
diagnosis_order[both_dates_present & (days_gp_minus_diabetes > 0)] = "diabetes_first"
diagnosis_order[both_dates_present & (days_gp_minus_diabetes == 0)] = "same_day"
df["E10_E11_vs_gastroparesis_order"] = diagnosis_order

print("\nDiabetes-vs-gastroparesis diagnosis order within the primary cohort:")
print(diagnosis_order[primary_cohort].value_counts())

# Make the boolean flag columns explicitly False rather than NaN before
# saving, so downstream filtering in Excel/pandas doesn't need to handle
# NaN-as-missing vs NaN-as-false ambiguity.
for col in [
    "meets_rao_LITERAL_gp_criteria", "meets_rao_adapted_gp_criteria",
    "meets_5yr_rule_E10_E11", "meets_1yr_concurrency_rule", "meets_5yr_concurrency_rule",
]:
    if col in df.columns:
        df[col] = df[col].fillna(False)

# --- Write output: only the primary cohort's rows, with every flag attached ---
output_df = df[primary_cohort].copy()

print("\nFinal cohort QA (every row should show only the True/before-surgery side):")
print(output_df["E10_E11_diabetes_before_surgery"].value_counts())
print(output_df["surgery_after_K31_84_dx"].value_counts())

# HARD VALIDATION, not a soft warning: first_diabetes_dx_date was built as
# the MINIMUM across the full E08-E13 block, which includes E10/E11 as a
# subset - so this should be mathematically guaranteed, not just "usually"
# true. If it fails, the two date columns were built with inconsistent
# logic somewhere upstream, which would make EVERY number in this script
# (and likely others using these same columns) unreliable. Positioned here,
# after the other diagnostics have already printed for debugging context,
# but BEFORE the file is written - a bad/inconsistent file should never be
# silently saved.
if n_e08e13_not > 0:
    raise ValueError(
        f"{n_e08e13_not:,} patients have E10/E11 diabetes before surgery but NOT E08-E13 "
        f"diabetes before surgery - this should be mathematically impossible if both date "
        f"columns were built consistently. Check diabetes date construction before trusting "
        f"any output from this script. NOT writing the output file."
    )

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(output_df):,} rows, {len(output_df.columns)} columns)")
