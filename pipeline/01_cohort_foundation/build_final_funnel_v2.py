"""
build_final_funnel_v2.py

Combines the now-confirmed-correct gating criteria, all from data already
on hand - no new scan needed for this step:

  1. Has bariatric surgery (2,842)
  2. K31.84 strictly within (Oct 1 2015, surgery_date) - using FIRST
     surgery only (bariatric_date), per policy: if first surgery doesn't
     qualify, patient is excluded outright, no checking later surgeries.
     Already scanned: has_K3184_strictly_in_window.
  3. E10 or E11 strictly before surgery (first occurrence is mathematically
     sufficient for this single-bound check - already in the master file,
     no rescan needed).

Age >=18 at surgery is NOT applied yet in this step - that needs a fresh
scan of patient.csv restricted to whoever survives steps 1-3, since the
existing age data was scoped to the OLD 1,070-patient population.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype

print(">>> SCRIPT VERSION: build_final_funnel_v2 <<<")

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
K3184_WINDOW_CSV = "bariatric_patients_K3184_window_check.csv"
E10E11_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

step1 = df["has_bariatric_surgery"]
print(f"Step 1 - has bariatric surgery: {step1.sum():,}")

# Merge in the K31.84 window check result
window_df = pd.read_csv(K3184_WINDOW_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "has_K3184_strictly_in_window", "latest_K3184_in_window"])
overlap = set(df.columns).intersection(window_df.columns) - {"patient_id"}
if overlap:
    raise ValueError(f"Merge would create _x/_y duplicate columns: {overlap}")
df = df.merge(window_df, on="patient_id", how="left")

# Scoped specifically to has_bariatric_surgery patients - checking against
# the full 335,846 would show ~333,000 "missing" entries that are expected
# (non-surgery patients were never in the window scan to begin with).
missing_window = (step1 & df["has_K3184_strictly_in_window"].isna()).sum()
print(f"Missing K31.84 window results among bariatric-surgery patients (should be 0): {missing_window:,}")

if not is_bool_dtype(df["has_K3184_strictly_in_window"]):
    df["has_K3184_strictly_in_window"] = df["has_K3184_strictly_in_window"].fillna(False).astype(bool)

step2 = step1 & df["has_K3184_strictly_in_window"]
print(f"Step 2 - + K31.84 strictly within (Oct 2015, first surgery date): {step2.sum():,}")

# Merge in E10/E11 dates
e10e11_df = pd.read_csv(E10E11_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "first_E10_date", "first_E11_date"])
overlap2 = set(df.columns).intersection(e10e11_df.columns) - {"patient_id"}
if overlap2:
    raise ValueError(f"Merge would create _x/_y duplicate columns: {overlap2}")
df = df.merge(e10e11_df, on="patient_id", how="left")

missing_diabetes = step1 & df["first_E10_date"].isna() & df["first_E11_date"].isna()
print(f"Bariatric-surgery patients missing BOTH E10 and E11 dates (no diabetes record at all): {missing_diabetes.sum():,}")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
e10_dt = pd.to_datetime(df["first_E10_date"], errors="coerce")
e11_dt = pd.to_datetime(df["first_E11_date"], errors="coerce")

e10_before = e10_dt.notna() & surgery_dt.notna() & (e10_dt < surgery_dt)
e11_before = e11_dt.notna() & surgery_dt.notna() & (e11_dt < surgery_dt)
diabetes_before_surgery = e10_before | e11_before
df["E10_E11_strictly_before_surgery"] = diabetes_before_surgery

step3 = step2 & diabetes_before_surgery
print(f"Step 3 - + E10/E11 strictly before surgery:                      {step3.sum():,}")

print("\n" + "="*70)
print(f"COHORT BEFORE AGE FILTER: {step3.sum():,}")
print("="*70)
print("\n(Age >=18 at surgery still needs to be applied - requires a fresh")
print(" patient.csv scan restricted to these survivors, since the existing")
print(" age data was scoped to the old 1,070-patient population.)")

df["passes_K3184_E10E11_gating"] = step3

# Hard checks: mathematically guaranteed by how step3 was built (it's an AND
# chain that already requires both components), but verifying explicitly
# rather than just trusting the construction - cheap insurance.
bad_k3184 = df[df["passes_K3184_E10E11_gating"] & ~df["has_K3184_strictly_in_window"]]
if len(bad_k3184) > 0:
    raise ValueError(f"Logic error: {len(bad_k3184)} patients passed gating without satisfying the K31.84 window check.")

bad_diabetes = df[df["passes_K3184_E10E11_gating"] & ~df["E10_E11_strictly_before_surgery"]]
if len(bad_diabetes) > 0:
    raise ValueError(f"Logic error: {len(bad_diabetes)} patients passed gating without diabetes strictly before surgery.")

print("\nHard consistency checks passed - no patient bypassed either underlying requirement.")

df.to_csv("funnel_v2_before_age.csv", index=False)
print(f"\nWrote funnel_v2_before_age.csv ({len(df):,} rows, includes the passes_K3184_E10E11_gating flag)")
