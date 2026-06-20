"""
verify_bariatric_date_meaning.py

For the ~102 multi-surgery patients, scans procedure.csv directly to find
ALL their actual bariatric surgery dates, then compares against the
bariatric_date value already stored in the master file - to determine
definitively whether bariatric_date represents the FIRST surgery, the
LAST surgery, or something else entirely.

This matters because surgery-after-diagnosis logic throughout this entire
project has used bariatric_date as if it's a single, unambiguous value -
if it's actually "first surgery" for multi-surgery patients, a later
surgery that DID occur after diagnosis could be getting missed.

Small, fast scan - restricted to ~102 patient_ids only.
"""

import subprocess
import pandas as pd
from pandas.api.types import is_bool_dtype

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"
BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"

BARIATRIC_CPT_CODES = {"43775", "43644", "43645", "43846", "43847"}

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

multi_surgery_ids = set(df[df["has_bariatric_surgery"] & (df["num_distinct_bariatric_surgery_dates"] > 1)]["patient_id"])
print(f"Multi-surgery patients to check: {len(multi_surgery_ids):,}")

stored_dates = dict(zip(df["patient_id"], df["bariatric_date"]))

print("\nScanning procedure.csv for ALL bariatric surgery dates for these patients...")
proc = subprocess.Popen(["gsutil", "cat", PROCEDURE_FILE], stdout=subprocess.PIPE)
all_dates = {}
rows_seen = 0
for chunk in pd.read_csv(proc.stdout, usecols=["patient_id", "code", "date"], dtype=str, chunksize=500_000):
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(multi_surgery_ids) & chunk["code"].isin(BARIATRIC_CPT_CODES)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y%m%d", errors="coerce")
        for pid, dates in chunk.groupby("patient_id")["date"]:
            all_dates.setdefault(pid, []).extend(dates.dropna().tolist())
proc.stdout.close()
proc.wait()
print(f"  scanned {rows_seen:,} rows")

print("\nComparing stored bariatric_date against actual min/max surgery dates per patient:")
n_matches_min = 0
n_matches_max = 0
n_matches_neither = 0
for pid, dates in all_dates.items():
    if not dates:
        continue
    actual_min = min(dates)
    actual_max = max(dates)
    stored = pd.to_datetime(stored_dates.get(pid), errors="coerce")
    if pd.isna(stored):
        continue
    if stored == actual_min:
        n_matches_min += 1
    elif stored == actual_max:
        n_matches_max += 1
    else:
        n_matches_neither += 1

print(f"  bariatric_date matches the EARLIEST surgery date: {n_matches_min:,}")
print(f"  bariatric_date matches the LATEST surgery date:   {n_matches_max:,}")
print(f"  bariatric_date matches NEITHER (something else):  {n_matches_neither:,}")
