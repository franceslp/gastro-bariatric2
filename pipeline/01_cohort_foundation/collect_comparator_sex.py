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
header_raw = proc.stdout.readline().decode().strip().split(",")
proc.stdout.close(); proc.wait()
# Strip surrounding quotes/whitespace — patient.csv uses quoted headers
header = [h.strip().strip('"').strip("'") for h in header_raw]
print(f"patient.csv columns: {header}")

sex_col = None
for cand in ["sex", "gender", "sex_at_birth"]:
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

import numpy as np
out = pd.DataFrame({"patient_id": list(comp_ids)})
out["sex_raw"] = out["patient_id"].map(sex_lookup)

# Encode F=1, M=0, but PRESERVE missing as NaN (do not silently code missing as male).
# Missing-sex patients will be dropped in complete-case analysis like any other
# missing covariate.
def encode_sex(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().upper()
    if s == "" or s in ("NAN", "NONE", "UNKNOWN", "U"):
        return np.nan
    if s[0] == "F":
        return 1
    if s[0] == "M":
        return 0
    return np.nan

out["sex_encoded"] = out["sex_raw"].map(encode_sex)

# QA
n_missing = out["sex_encoded"].isna().sum()
n_f = (out["sex_encoded"] == 1).sum()
n_m = (out["sex_encoded"] == 0).sum()
print(f"sex_encoded F(1): {n_f} | M(0): {n_m} | missing(NaN): {n_missing}")
print(f"F% of non-missing: {100*n_f/(n_f+n_m):.1f}%")
out[["patient_id", "sex_encoded"]].to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV}: {len(out)} rows")
