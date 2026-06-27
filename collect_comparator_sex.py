#!/usr/bin/env python3
"""
collect_comparator_sex.py

Quick targeted fix: the comparator demographics script collected race and
ethnicity from patient.csv but NOT sex. sex_encoded is a PSM covariate, so
we need it. This scans patient.csv (small, ~2M rows) for the comparator
cohort's sex only.

INPUT:  comparator_pool_ready_for_PSM.csv (patient_ids)
        gs://.../patient.csv (sex)
OUTPUT: comparator_sex.csv  (patient_id, sex_encoded)

sex_encoded: F=1, M=0 (matches GP encoding in assemble_and_run_psm.py)
"""
import subprocess
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PATIENT_FILE = f"{GCS_BASE}/patient.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
OUTPUT_CSV = "comparator_sex.csv"
CHUNK = 500_000

comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str})
comp_ids = set(comp["patient_id"].dropna())
print(f"Comparator patients: {len(comp_ids):,}")

def stream(path, usecols):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

# Detect sex column name first
import io
head = subprocess.run(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
# read just header via a quick separate small read
proc = subprocess.Popen(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
header = proc.stdout.readline().decode().strip().split(",")
proc.stdout.close(); proc.wait()
print(f"patient.csv columns: {header}")

sex_col = None
for cand in ["sex", "gender", "sex_at_birth", "Sex", "Gender"]:
    if cand in header:
        sex_col = cand
        break
if sex_col is None:
    raise SystemExit(f"No sex column found in patient.csv header: {header}")
print(f"Using sex column: '{sex_col}'")

sex_lookup = {}
rows = 0
for chunk in stream(PATIENT_FILE, usecols=["patient_id", sex_col]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(comp_ids)]
    for pid, sx in zip(chunk["patient_id"], chunk[sex_col]):
        if pid not in sex_lookup:
            sex_lookup[pid] = sx
print(f"Scanned {rows:,} patient rows; found sex for {len(sex_lookup):,}/{len(comp_ids):,}")

out = pd.DataFrame({"patient_id": list(comp_ids)})
out["sex_raw"] = out["patient_id"].map(sex_lookup)
out["sex_encoded"] = (out["sex_raw"].astype(str).str.upper().str[0] == "F").astype(int)

# QA
print(f"sex_encoded F(1): {out['sex_encoded'].sum()} ({100*out['sex_encoded'].mean():.1f}%)")
print(f"missing sex: {out['sex_raw'].isna().sum()}")
out[["patient_id", "sex_encoded"]].to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV}: {len(out)} rows")
