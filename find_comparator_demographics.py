"""
find_comparator_demographics.py

STAGE 3 of 3 for building the PSM comparator group.

For the comparator candidates from Stage 2, scans patient.csv to pull:
  - year_of_birth (to compute age at surgery)
  - sex

Then:
  - Computes age_at_surgery_approx = surgery_year - year_of_birth
  - Applies age >= 18 filter
  - Adds a binary surgery_type covariate (sleeve vs bypass) for PSM

After this script, the comparator pool is ready for PSM against the
110-patient gastroparesis cohort.

PSM covariates that will be used:
  - age_at_surgery_approx (continuous)
  - sex (categorical)
  - surgery_type (sleeve vs bypass, binary)
  - has_diabetes (binary)
  - surgery_year (exact year)
"""

import subprocess
import pandas as pd

PATIENT_FILE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia/patient.csv"
INPUT_CSV = "comparator_with_covariates.csv"
OUTPUT_CSV = "comparator_pool_ready_for_PSM.csv"

SLEEVE_CODES = {"43775"}
BYPASS_CODES = {"43644", "43645", "43846", "43847"}

print(">>> SCRIPT VERSION: find_comparator_demographics_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"].dropna())
print(f"Stage 2 candidates: {len(cohort_ids):,}")

print(f"\nScanning patient.csv for year_of_birth and sex...")
proc = subprocess.Popen(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
demo_records = {}
duplicate_rows = 0
rows_seen = 0
for chunk in pd.read_csv(proc.stdout,
                          usecols=["patient_id", "year_of_birth", "sex"],
                          dtype=str, chunksize=500_000):
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        for _, row in chunk.iterrows():
            pid = row["patient_id"]
            if pid in demo_records:
                duplicate_rows += 1
            else:
                demo_records[pid] = {"year_of_birth": row["year_of_birth"],
                                      "sex": row["sex"]}
proc.stdout.close()
proc.wait()

print(f"  done - scanned {rows_seen:,} rows")
print(f"  matched: {len(demo_records):,}/{len(cohort_ids):,}")
print(f"  missing demographics: {len(cohort_ids)-len(demo_records):,}")
print(f"  duplicate rows in patient.csv: {duplicate_rows:,}")

demo_df = pd.DataFrame.from_dict(demo_records, orient="index").reset_index()
demo_df.columns = ["patient_id", "year_of_birth", "sex"]

df = df.merge(demo_df, on="patient_id", how="left")
assert len(df) == len(cohort_ids), f"Row count changed after merge: {len(df)}"

# Compute age at surgery
surgery_year = pd.to_datetime(df["bariatric_date"], errors="coerce").dt.year
df["year_of_birth"] = pd.to_numeric(df["year_of_birth"], errors="coerce")
df["age_at_surgery_approx"] = surgery_year - df["year_of_birth"]

# Age >= 18 filter
n_before = len(df)
df = df[df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] >= 18)].copy()
n_after = len(df)
print(f"\nAge >= 18 filter: {n_before:,} -> {n_after:,} (removed {n_before-n_after:,})")

# Surgery type covariate for PSM
def surgery_type(codes_str):
    if pd.isna(codes_str):
        return None
    codes = set(codes_str.split(","))
    has_sleeve = bool(codes & SLEEVE_CODES)
    has_bypass = bool(codes & BYPASS_CODES)
    if has_sleeve and has_bypass:
        # Both sleeve and bypass CPTs on same date - ambiguous (could be
        # revision, conversion, or billing artifact). Excluding rather than
        # arbitrarily assigning one type - cleaner for PSM and defensible
        # in methods as "ambiguous procedure coding excluded."
        return None
    elif has_sleeve:
        return "sleeve"
    elif has_bypass:
        return "bypass"
    return None

df["surgery_type"] = df["bariatric_cpt_codes_seen"].apply(surgery_type)

# Drop missing sex - sex is a core PSM covariate, can't match without it
n_missing_sex = df["sex"].isna().sum()
print(f"Dropping {n_missing_sex:,} patients with missing sex")
df = df[df["sex"].notna()].copy()

# Drop unknown surgery type - can't match on surgery type without knowing it
n_before_type = len(df)
n_missing_type = df["surgery_type"].isna().sum()
print(f"Dropping ambiguous/unknown surgery type: "
      f"{n_missing_type:,}/{n_before_type:,} ({100*n_missing_type/n_before_type:.1f}%)")
df = df[df["surgery_type"].notna()].copy()

# NOTE: BMI is not currently included as a PSM covariate but should be
# discussed with Dr. Sujka - bariatric outcomes are strongly related to
# baseline BMI, and if the study question is "does gastroparesis affect
# A1c response after bariatric surgery," matching on BMI makes sense.
# If BMI is considered a mediator of surgical outcome rather than a
# confounder, including it in PSM could over-adjust. PI decision needed.

# Age as regular int (not nullable Int64) - missing ages already dropped
# above, so no NaNs remain. Regular int avoids downstream compatibility
# issues with scikit-learn and statsmodels.
df["age_at_surgery_approx"] = df["age_at_surgery_approx"].round().astype(int)

# Final PSM covariate summary
print(f"\nFinal comparator pool: {len(df):,} patients")
print(f"\nPSM covariate summary:")
print(f"\nSex:")
print(df["sex"].value_counts(dropna=False))
print(f"\nSurgery type:")
print(df["surgery_type"].value_counts())
print(f"\nHas diabetes:")
print(df["has_diabetes"].value_counts())
print(f"\nAge distribution:")
print(df["age_at_surgery_approx"].describe())
print(f"\nSurgery year distribution:")
print(df["surgery_year"].value_counts().sort_index())

# Missing covariate check - PSM needs complete cases
print(f"\nMissing values per PSM covariate:")
for col in ["age_at_surgery_approx", "sex", "surgery_type", "has_diabetes", "surgery_year"]:
    n_missing = df[col].isna().sum() if col != "surgery_type" else (df[col] == "unknown").sum()
    print(f"  {col}: {n_missing:,} missing/unknown")

assert df["patient_id"].is_unique, "Duplicate patient_ids in final pool"

# Final completeness check - drop any remaining incomplete PSM records
PSM_VARS = ["age_at_surgery_approx", "sex", "surgery_type", "has_diabetes", "surgery_year"]
n_before_drop = len(df)
df = df.dropna(subset=PSM_VARS)
n_removed = n_before_drop - len(df)
print(f"\nRemoved incomplete PSM records (missing any covariate): {n_removed:,}")
print(f"Final PSM-ready comparator pool: {len(df):,} patients")

df["comparator_stage"] = "stage3_demographics_added"

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(df):,} rows)")
print("Ready for PSM against the 110-patient gastroparesis cohort.")
