"""
verify_primary_cohort_clean.py

CLEAN VERIFICATION - only the 4 core criteria, nothing else. No Rao, no
Definition 1, no extra informational flags. This exists specifically to
confirm the 1,070 number is correct and reproducible, with the funnel
shown step by step so each criterion's effect is visible on its own.

The 4 criteria, all required together (AND):
  1. K31.84 diagnosed on/after Oct 1, 2015 (in_study_period)
  2. Has a qualifying bariatric surgery code (has_bariatric_surgery)
  3. Surgery strictly AFTER the K31.84 diagnosis (bariatric_date > first_K31_84_date)
  4. E10 or E11 diabetes diagnosed strictly BEFORE surgery (no time window cap)

No new GCS scan - merges two existing files only.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype

print(">>> SCRIPT VERSION: verify_primary_cohort_clean_v1 <<<")

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
E10E11_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"\nStarting population (full gastroparesis cohort, K31.84 or 536.3, any year): {len(df):,}")

e10e11_df = pd.read_csv(E10E11_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "first_E10_date", "first_E11_date"])

overlap = set(df.columns).intersection(e10e11_df.columns) - {"patient_id"}
print(f"Column overlap before merge (should be empty): {overlap}")
if overlap:
    raise ValueError(f"Merge would create _x/_y duplicate columns: {overlap}. Stopping before merging.")

df = df.merge(e10e11_df, on="patient_id", how="left")

n_unique = df["patient_id"].nunique()
if n_unique != len(df):
    raise ValueError(f"Merge produced {len(df):,} rows but only {n_unique:,} unique patient_ids - duplicate rows in an input file.")

# Confirm the merge actually attached diabetes dates - a failed/empty merge
# would silently turn everyone into "no diabetes" further down, without
# raising any error, since notna() on an all-missing column just returns
# all False rather than crashing.
print("\nDiabetes date attachment QA:")
print(f"  Missing E10 dates: {df['first_E10_date'].isna().sum():,}/{len(df):,}")
print(f"  Missing E11 dates: {df['first_E11_date'].isna().sum():,}/{len(df):,}")
print(f"  Patients with either E10 or E11 (any time): {(df['first_E10_date'].notna() | df['first_E11_date'].notna()).sum():,}")

if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")
if not is_bool_dtype(df["in_study_period"]):
    df["in_study_period"] = df["in_study_period"].astype(str).str.strip().str.lower().eq("true")

dx_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
e10_dt = pd.to_datetime(df["first_E10_date"], errors="coerce")
e11_dt = pd.to_datetime(df["first_E11_date"], errors="coerce")

# --- Step 1: in_study_period ---
step1 = df["in_study_period"]
print(f"\nStep 1 - K31.84 on/after Oct 1, 2015 (in_study_period): {step1.sum():,}")

# --- Step 2: + has_bariatric_surgery ---
step2 = step1 & df["has_bariatric_surgery"]
print(f"Step 2 - + has a qualifying bariatric surgery code:      {step2.sum():,}")

# --- Step 3: + surgery strictly after K31.84 dx ---
surgery_after_dx = dx_dt.notna() & surgery_dt.notna() & (surgery_dt > dx_dt)
step3 = step2 & surgery_after_dx
print(f"Step 3 - + surgery strictly AFTER K31.84 diagnosis:      {step3.sum():,}")

# --- Step 4: + E10 or E11 diabetes strictly before surgery ---
e10_before_surgery = e10_dt.notna() & surgery_dt.notna() & (e10_dt < surgery_dt)
e11_before_surgery = e11_dt.notna() & surgery_dt.notna() & (e11_dt < surgery_dt)
diabetes_before_surgery = e10_before_surgery | e11_before_surgery
step4 = step3 & diabetes_before_surgery
print(f"Step 4 - + E10/E11 diabetes strictly BEFORE surgery:     {step4.sum():,}")

print("\n" + "="*70)
print(f"FINAL PRIMARY COHORT: {step4.sum():,}")
print("="*70)

if step4.sum() == 1070:
    print("\nMatches the previously established 1,070 - confirmed reproducible.")
else:
    print(f"\nDOES NOT MATCH the previously established 1,070 (got {step4.sum():,} instead) -")
    print("something changed. Investigate before trusting any downstream work.")
