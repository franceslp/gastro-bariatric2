"""
check_5yr_concurrency_E10_E11_simple.py

Simple question: of the patients who had bariatric surgery, how many also
had BOTH a gastroparesis diagnosis (K31.84) AND a Type 1 or Type 2 diabetes
diagnosis (E10 or E11) documented within 5 years before their surgery date?

This mirrors Sadda et al.'s concurrency rule but uses E10/E11 only instead
of the full E08-E13 diabetes block.

The 5-year rule: max(first_K31_84_date, first_E10_or_E11_date) must fall
within 5 years (1826 days) before the surgery date. Neither diagnosis has
to come first - either order qualifies.

Requires one scan of diagnosis.csv to get first E10/E11 dates (not already
in the master file). Everything else already exists in the master file.
"""

import os
import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"

WINDOW_DAYS = round(5 * 365.25)  # 1826 days

print(f"Loading {INPUT_CSV}...")
full_df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

# Work only with the bariatric surgery subset
from pandas.api.types import is_bool_dtype
if not is_bool_dtype(full_df["has_bariatric_surgery"]):
    full_df["has_bariatric_surgery"] = (
        full_df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")
    )
df = full_df[full_df["has_bariatric_surgery"]].copy()
bariatric_ids = set(df["patient_id"])
print(f"  bariatric surgery subset: {len(bariatric_ids):,} patients")
print(f"  of those, in-study-period K31.84: {df['in_study_period'].sum():,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for earliest E10/E11 codes, restricted to {len(bariatric_ids):,} patients...")

first_E10_date = {}
first_E11_date = {}
rows_seen = 0
chunk_num = 0

for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(bariatric_ids)]
    if not chunk.empty:
        icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
        if not icd.empty:
            icd["date"] = pd.to_datetime(icd["date"], format="%Y%m%d", errors="coerce")

            e10 = icd[icd["code"].str.startswith("E10", na=False)]
            if not e10.empty:
                for pid, d in e10.groupby("patient_id")["date"].min().items():
                    if pd.notna(d) and (pid not in first_E10_date or d < first_E10_date[pid]):
                        first_E10_date[pid] = d

            e11 = icd[icd["code"].str.startswith("E11", na=False)]
            if not e11.empty:
                for pid, d in e11.groupby("patient_id")["date"].min().items():
                    if pd.notna(d) and (pid not in first_E11_date or d < first_E11_date[pid]):
                        first_E11_date[pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"  done - scanned {rows_seen:,} rows")

# --- Apply 5-year concurrency rule ---
df["first_E10_date"] = df["patient_id"].map(first_E10_date)
df["first_E11_date"] = df["patient_id"].map(first_E11_date)

first_E10_dt = pd.to_datetime(df["first_E10_date"], errors="coerce")
first_E11_dt = pd.to_datetime(df["first_E11_date"], errors="coerce")
first_K31_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
surgery_dt   = pd.to_datetime(df["bariatric_date"], errors="coerce")

# Earliest of E10 or E11
first_diabetes_dt = pd.concat([first_E10_dt, first_E11_dt], axis=1).min(axis=1, skipna=True)
df["first_E10_or_E11_date"] = first_diabetes_dt

# Both diagnoses must be present
both_present = first_diabetes_dt.notna() & first_K31_dt.notna() & surgery_dt.notna()

# Concurrent date = the LATER of the two diagnoses
concurrent_dt = pd.concat([first_K31_dt, first_diabetes_dt], axis=1).max(axis=1)
concurrent_dt = concurrent_dt.where(both_present, pd.NaT)
df["concurrent_date_E10_E11"] = concurrent_dt

days_to_surgery = (surgery_dt - concurrent_dt).dt.days
df["days_concurrent_E10_E11_to_surgery"] = days_to_surgery

df["meets_5yr_rule_E10_E11"] = (
    both_present
    & (days_to_surgery >= 0)
    & (days_to_surgery <= WINDOW_DAYS)
)

# --- Results ---
in_period = df["in_study_period"]
n = in_period.sum()
print(f"\nOf the {n:,} in-study-period K31.84 + bariatric surgery patients:")
print(f"  have E10 or E11 code (any time):          {(in_period & first_diabetes_dt.notna()).sum():,}")
print(f"  meet 5yr concurrency rule (E10/E11 only): {(in_period & df['meets_5yr_rule_E10_E11']).sum():,}")
if "meets_5yr_concurrency_rule" in df.columns:
    print(f"  (for reference) 5yr rule E08-E13 (Sadda): {(in_period & df['meets_5yr_concurrency_rule']).sum():,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time() - SCRIPT_START_TIME) / 60:.1f} minutes")
