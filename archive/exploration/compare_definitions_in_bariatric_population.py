"""
compare_definitions_in_bariatric_population.py

Dr. Sujka's question: of the two candidate gastroparesis definitions, which
gives the LARGER population once we also require bariatric surgery? (Not
just the larger population in the full 263,087-patient base cohort - the
actual bariatric-surgery comparator group is what matters here.)

Definition 1: K31.84 + prokinetic (after dx) + GES (before/on dx) - 19,641
              patients in the full base cohort
Definition 2: Rao adapted criteria (GES timing + symptom + endoscopy) -
              2,408 (literal) or 3,904 (same-day-modified) in the full base
              cohort

No new scan needed - this just merges two existing output files on
patient_id and filters to the bariatric-surgery subset:
  - gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv
    (has has_bariatric_surgery, bariatric_date, first_GES_date,
    first_K31_84_date, any_prokinetic_ever_after_dx)
  - gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv
    (has meets_rao_LITERAL_gp_criteria, meets_rao_adapted_gp_criteria)
"""

import pandas as pd

print(">>> SCRIPT VERSION: compare_definitions_in_bariatric_population_v1 <<<")

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
RAO_CSV = "gastroparesis_prokinetic_cohort_with_rao_literal_and_modified.csv"

bariatric_df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
rao_df = pd.read_csv(RAO_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "meets_rao_LITERAL_gp_criteria", "meets_rao_adapted_gp_criteria"])

df = bariatric_df.merge(rao_df, on="patient_id", how="left")
print(f"Merged: {len(df):,} total patients in the combined file\n")

from pandas.api.types import is_bool_dtype
if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

in_period_bariatric = df["in_study_period"] & df["has_bariatric_surgery"]
print(f"In-study-period K31.84 patients WITH bariatric surgery: {in_period_bariatric.sum():,}\n")

# All "base cohort" counts below are restricted to in_study_period, to match
# the denominator used everywhere else in this project (263,087) - without
# this, out-of-period K31.84 patients (diagnosed before Oct 2015) who still
# happen to satisfy the other criteria would silently inflate these numbers.
base_period = df["in_study_period"]

# --- Definition 1: prokinetic (after dx) + GES (before/on dx) ---
ges_dt = pd.to_datetime(df["first_GES_date"], errors="coerce")
dx_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_before_or_same_day = ges_dt.notna() & dx_dt.notna() & (ges_dt <= dx_dt)
def1_met = df["any_prokinetic_ever_after_dx"].fillna(False) & ges_before_or_same_day

n_def1_base = (def1_met & base_period).sum()
n_def1_bariatric = (def1_met & in_period_bariatric).sum()

print("DEFINITION 1: K31.84 + prokinetic (after dx) + GES (before/on dx)")
print(f"  full base cohort (in-study-period):        {n_def1_base:,}")
print(f"  AND has bariatric surgery: {n_def1_bariatric:,}")

# --- Definition 2: Rao criteria ---
rao_literal_met = df["meets_rao_LITERAL_gp_criteria"].fillna(False)
rao_modified_met = df["meets_rao_adapted_gp_criteria"].fillna(False)

n_rao_lit_base = (rao_literal_met & base_period).sum()
n_rao_lit_bariatric = (rao_literal_met & in_period_bariatric).sum()
n_rao_mod_base = (rao_modified_met & base_period).sum()
n_rao_mod_bariatric = (rao_modified_met & in_period_bariatric).sum()

print("\nDEFINITION 2a: Rao criteria, LITERAL (published 7-90 day GES window)")
print(f"  full base cohort (in-study-period):        {n_rao_lit_base:,}")
print(f"  AND has bariatric surgery: {n_rao_lit_bariatric:,}")

print("\nDEFINITION 2b: Rao criteria, MODIFIED (same-day GES allowed)")
print(f"  full base cohort (in-study-period):        {n_rao_mod_base:,}")
print(f"  AND has bariatric surgery: {n_rao_mod_bariatric:,}")

print("\n--- SUMMARY: bariatric-surgery population only ---")
print(f"  Definition 1 (prokinetic + GES):  {n_def1_bariatric:,}")
print(f"  Definition 2a (Rao, literal):      {n_rao_lit_bariatric:,}")
print(f"  Definition 2b (Rao, modified):     {n_rao_mod_bariatric:,}")

# --- Overlap: how many patients satisfy BOTH definitions? ---
# If overlap is high relative to the smaller group, the two definitions are
# largely identifying the same patients (just one is more permissive). If
# overlap is low, they're capturing genuinely different phenotypes - worth
# knowing regardless of which one ends up chosen, since it affects how
# defensible "Definition X is just a looser version of Rao" would be as a
# claim in a methods section.
overlap_def1_rao_modified = (def1_met & rao_modified_met & in_period_bariatric).sum()
overlap_def1_rao_literal = (def1_met & rao_literal_met & in_period_bariatric).sum()

print("\n--- OVERLAP (within bariatric-surgery population) ---")
print(f"  Definition 1 AND Rao literal:   {overlap_def1_rao_literal:,}")
print(f"  Definition 1 AND Rao modified:  {overlap_def1_rao_modified:,}")

if n_rao_mod_bariatric > 0:
    print(
        f"  Percent of Rao-modified patients also meeting Definition 1: "
        f"{100 * overlap_def1_rao_modified / n_rao_mod_bariatric:.1f}%"
    )
if n_def1_bariatric > 0:
    print(
        f"  Percent of Definition 1 patients also meeting Rao modified: "
        f"{100 * overlap_def1_rao_modified / n_def1_bariatric:.1f}%"
    )
print("  (A high first number + low second number means Definition 1 is the broader")
print("   net and largely contains Rao-modified patients within it, rather than the two")
print("   capturing genuinely separate phenotypes.)")
