"""
find_comparator_diagnosis_covariates.py

STAGE 2 of 3 for building the PSM comparator group.

For the comparator candidates from Stage 1, scans diagnosis.csv to:
  1. Confirm no K31.84 or 536.3 gastroparesis codes (safety check)
  2. Capture E10/E11 diabetes status (PSM covariate)
  3. Derive surgery year from bariatric_date (PSM covariate per Sadda)

Patients with any gastroparesis code found are removed from the pool.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "comparator_bariatric_candidates.csv"
OUTPUT_CSV = "comparator_with_covariates.csv"

GP_CODES = {"K31.84", "536.3"}

print(">>> SCRIPT VERSION: find_comparator_diagnosis_covariates_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"].dropna())
print(f"Stage 1 candidates: {len(cohort_ids):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for gastroparesis codes and diabetes status...")

has_gastroparesis = set()  # to EXCLUDE
has_E10 = set()
has_E11 = set()

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()
        icd = chunk[chunk["code_system"].isin(["ICD-10-CM", "ICD-9-CM"])]

        # Gastroparesis check - both modern and legacy codes
        gp_hits = icd[icd["code"].isin(GP_CODES)]
        if not gp_hits.empty:
            has_gastroparesis.update(gp_hits["patient_id"].unique())

        # E10/E11 diabetes status
        icd10 = icd[icd["code_system"] == "ICD-10-CM"]
        e10_hits = icd10[icd10["code"].str.startswith("E10", na=False)]
        if not e10_hits.empty:
            has_E10.update(e10_hits["patient_id"].unique())

        e11_hits = icd10[icd10["code"].str.startswith("E11", na=False)]
        if not e11_hits.empty:
            has_E11.update(e11_hits["patient_id"].unique())

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, "
              f"{len(has_gastroparesis):,} GP found, {len(has_E10|has_E11):,} diabetic found)")

print(f"\n  done - scanned {rows_seen:,} rows")
print(f"  Patients with gastroparesis codes found (to exclude): {len(has_gastroparesis):,}")
print(f"  Patients with E10 (Type 1 diabetes): {len(has_E10):,}")
print(f"  Patients with E11 (Type 2 diabetes): {len(has_E11):,}")
print(f"  Patients with both E10 and E11: {len(has_E10 & has_E11):,}")
print(f"  Patients with any diabetes (E10 or E11): {len(has_E10 | has_E11):,}")

# Apply gastroparesis exclusion
df = df[~df["patient_id"].isin(has_gastroparesis)].copy()
print(f"\nAfter gastroparesis exclusion: {len(df):,} candidates remaining")
print(f"  Removed for GP codes: {len(has_gastroparesis):,}/{len(cohort_ids):,} "
      f"({100*len(has_gastroparesis)/len(cohort_ids):.1f}%)")

# Diabetes covariate - simplified to binary for PSM stability.
# "Both" (E10+E11) is rare enough that a three-way split would create
# sparse matching strata. Binary has_diabetes is cleaner and still
# captures the key confound for PSM against an unrestricted bariatric pool.
df["has_diabetes"] = df["patient_id"].isin(has_E10 | has_E11)

# Surgery year as PSM covariate - exact year, consistent with how
# the study group's surgery year will be computed from bariatric_date.
df["surgery_year"] = pd.to_datetime(df["bariatric_date"], errors="coerce").dt.year

n_missing_year = df["surgery_year"].isna().sum()
print(f"  Missing surgery year (date parsing failures): {n_missing_year:,}")

print("\nDiabetes status in comparator pool:")
print(df["has_diabetes"].value_counts())

print("\nSurgery year distribution:")
print(df["surgery_year"].value_counts().sort_index())

assert df["patient_id"].is_unique, "Duplicate patient_ids after exclusion"

# Stage metadata - helps identify file provenance months later
df["comparator_stage"] = "stage2_gp_excluded"

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(df):,} rows)")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
print("\nNext: run find_comparator_demographics.py (Stage 3) to add age/sex for PSM")
