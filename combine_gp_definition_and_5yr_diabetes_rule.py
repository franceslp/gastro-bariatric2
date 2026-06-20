"""
combine_gp_definition_and_5yr_diabetes_rule.py

The REAL final-cohort question: how many patients satisfy ALL THREE
requirements at once?
  1. A gastroparesis definition (Definition 1: prokinetic+GES, OR Rao
     literal, OR Rao modified)
  2. The 5-year E10/E11 diabetes-concurrency rule
  3. Bariatric surgery

Previous scripts only ever checked these requirements two at a time:
  - compare_definitions_in_bariatric_population.py: gp definition + surgery
    (no diabetes-timing requirement at all)
  - check_5yr_concurrency_E10_E11_simple.py: diabetes-timing + surgery
    (no gastroparesis-strictness requirement beyond plain K31.84)
Neither one combines all three. This script does, by merging the three
existing output files on patient_id - no new GCS scan needed.
"""

import pandas as pd

print(">>> SCRIPT VERSION: combine_gp_definition_and_5yr_diabetes_rule_v1 <<<")

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
RAO_CSV = "gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv"
E10E11_5YR_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"

bariatric_df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
rao_df = pd.read_csv(RAO_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "meets_rao_LITERAL_gp_criteria", "meets_rao_adapted_gp_criteria"])
e10e11_df = pd.read_csv(E10E11_5YR_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "meets_5yr_rule_E10_E11"])

df = bariatric_df.merge(rao_df, on="patient_id", how="left")
df = df.merge(e10e11_df, on="patient_id", how="left")
print(f"Merged: {len(df):,} total patients")

# Sanity check: if any input file had duplicate patient_id rows, the merge
# above would silently fan out into a cartesian product and inflate every
# count below. This catches that before it can happen unnoticed.
n_unique = df["patient_id"].nunique()
if n_unique != len(df):
    raise ValueError(
        f"Merge produced {len(df):,} rows but only {n_unique:,} unique patient_ids - "
        f"one of the three input files has duplicate patient_id rows. Fix that before "
        f"trusting any counts below."
    )
print(f"Unique patients: {n_unique:,} (matches row count - no duplication from the merge)\n")

from pandas.api.types import is_bool_dtype
if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

base_period = df["in_study_period"]
has_surgery = df["has_bariatric_surgery"].fillna(False)
meets_5yr = df["meets_5yr_rule_E10_E11"].fillna(False)

# --- Definition 1: prokinetic (after dx) + GES (before/on dx) ---
ges_dt = pd.to_datetime(df["first_GES_date"], errors="coerce")
dx_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_before_or_same_day = ges_dt.notna() & dx_dt.notna() & (ges_dt <= dx_dt)
# dx_dt.notna() is already enforced via ges_before_or_same_day above, but it's
# made explicit here too on purpose - defensive against a future refactor of
# ges_before_or_same_day accidentally dropping that check without anyone
# noticing def1_met silently depended on it.
def1_met = (
    df["first_K31_84_date"].notna()
    & df["any_prokinetic_ever_after_dx"].fillna(False)
    & ges_before_or_same_day
)

rao_literal_met = df["meets_rao_LITERAL_gp_criteria"].fillna(False)
rao_modified_met = df["meets_rao_adapted_gp_criteria"].fillna(False)

print("FINAL COHORT INTERSECTION - gastroparesis definition AND 5yr E10/E11 rule AND bariatric surgery:\n")

for label, gp_def_met in [
    ("Definition 1 (prokinetic + GES)", def1_met),
    ("Rao criteria, literal", rao_literal_met),
    ("Rao criteria, modified", rao_modified_met),
]:
    gp_only = (gp_def_met & base_period & has_surgery).sum()
    gp_and_5yr = (gp_def_met & base_period & has_surgery & meets_5yr).sum()
    print(f"{label}:")
    print(f"  + bariatric surgery (no diabetes-timing check):  {gp_only:,}")
    print(f"  + bariatric surgery + 5yr E10/E11 rule:          {gp_and_5yr:,}")
    print()

print("(For reference) plain K31.84 + bariatric surgery + 5yr E10/E11 rule, no gp-definition strictness applied:")
print(f"  {(base_period & has_surgery & meets_5yr).sum():,}")
