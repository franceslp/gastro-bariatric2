"""
collect_comparator_medications_and_dx.py

Collects 12 PSM covariates for the comparator pool using IDENTICAL
methodology to bariatric_step5_covariates.py (GP cohort source):

  Source: medication_ingredient.csv + RxNorm ingredient codes
  Window: lifetime pre-op (start_date < bariatric_date) — same as GP cohort

  Medications (9): metformin, any_insulin, rapid_insulin, long_insulin,
                   glp1, sglt2, dpp4, sulfonylurea, tzd
  DM complications (2): dm_circulatory, dm_other
  Comorbidity (1): dyslipidemia
"""

import subprocess
import time
import pandas as pd
from collections import defaultdict

GCS_BASE        = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MEDICATION_FILE = f"{GCS_BASE}/medication_ingredient.csv"
DIAGNOSIS_FILE  = f"{GCS_BASE}/diagnosis.csv"

COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
GP_COVARIATES  = "study_covariates.csv"
OUTPUT_CSV     = "psm_covariates_comparator_meds_dx.csv"

print(">>> SCRIPT VERSION: collect_comparator_medications_and_dx_v6 <<<")

# ---------------------------------------------------------------------------
# Load comparator pool
# ---------------------------------------------------------------------------
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str},
                   usecols=["patient_id", "bariatric_date"])
comp = comp.drop_duplicates(subset="patient_id")
comp["surgery_dt"] = pd.to_datetime(comp["bariatric_date"], errors="coerce")

surgery_lookup = dict(zip(comp["patient_id"], comp["surgery_dt"]))
comp_ids = set(comp["patient_id"].dropna())
print(f"Comparator patients: {len(comp_ids):,}")

# ---------------------------------------------------------------------------
# RxNorm ingredient codes — IDENTICAL to bariatric_step5_covariates.py
# ---------------------------------------------------------------------------
MED_CODES = {
    "metformin":     {"6809"},
    "rapid_insulin": {"51428", "86009", "311036", "1156706"},
    "long_insulin":  {"253182", "274783", "1151131", "2200801"},
    "glp1":          {"60548", "475968", "2200644", "1991302", "1440051"},
    "sglt2":         {"1488574", "1545653", "1602111", "1932591"},
    "dpp4":          {"593411", "593533", "1100699", "884220"},
    "sulfonylurea":  {"4815", "4821", "25789"},
    "tzd":           {"33738", "84108"},
}
# Build reverse lookup: code -> drug class
code_to_class = {}
for drug_class, codes in MED_CODES.items():
    for code in codes:
        code_to_class[code] = drug_class

# ---------------------------------------------------------------------------
# Diagnosis patterns — dm_other excludes E10.9/E11.9 (near-universal)
# ---------------------------------------------------------------------------
DX_PATTERNS = {
    "dm_circulatory": [
        "E10.5", "E11.5",
        "E10.51", "E10.52", "E10.59",
        "E11.51", "E11.52", "E11.59",
    ],
    "dm_other": [
        "E10.6", "E11.6",
        "E10.8", "E11.8",
    ],
    "dyslipidemia": ["E78"],
}

# ---------------------------------------------------------------------------
# Initialise result dicts
# ---------------------------------------------------------------------------
med_flags = {pid: defaultdict(int) for pid in comp_ids}
dx_flags  = {pid: {k: 0 for k in DX_PATTERNS} for pid in comp_ids}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def stream_gcs_csv(path, usecols, chunksize=1_000_000):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(
            proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize
        ):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

def parse_date_col(series):
    """Handle both YYYYMMDD and ISO date formats."""
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    return result

# ===========================================================================
# PASS 1 — medication_ingredient.csv (RxNorm codes)
# ===========================================================================
print("\n--- PASS 1: medication_ingredient.csv (RxNorm ingredient codes) ---")
t0 = time.time()
rows_seen = chunk_num = 0

for chunk in stream_gcs_csv(
    MEDICATION_FILE,
    usecols=["patient_id", "code", "start_date"],
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(comp_ids)].copy()
    if chunk.empty:
        continue

    # Keep only relevant RxNorm codes
    chunk = chunk[chunk["code"].isin(code_to_class)]
    if chunk.empty:
        continue

    chunk["start_date"] = parse_date_col(chunk["start_date"])
    chunk["surgery_dt"] = chunk["patient_id"].map(surgery_lookup)

    # Lifetime pre-op use (start_date < surgery_date) — same as GP cohort
    chunk = chunk[chunk["start_date"] < chunk["surgery_dt"]]
    if chunk.empty:
        continue

    chunk["drug_class"] = chunk["code"].map(code_to_class)
    chunk = chunk[chunk["drug_class"].notna()]
    for pid, drug_class in zip(chunk["patient_id"], chunk["drug_class"]):
        med_flags[pid][drug_class] = 1

    if chunk_num % 50 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  ...{rows_seen:,} rows | {elapsed:.1f} min")

print(f"Done medication_ingredient.csv: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

# ===========================================================================
# PASS 2 — diagnosis.csv
# ===========================================================================
print("\n--- PASS 2: diagnosis.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

all_prefixes = tuple(p for patterns in DX_PATTERNS.values() for p in patterns)

for chunk in stream_gcs_csv(
    DIAGNOSIS_FILE,
    usecols=["patient_id", "code_system", "code", "date"],
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(comp_ids)].copy()
    if chunk.empty:
        continue

    chunk["code_system"] = chunk["code_system"].str.strip()
    icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
    if icd.empty:
        continue

    icd["code"] = icd["code"].str.strip()
    icd = icd[icd["code"].str.startswith(all_prefixes)]
    if icd.empty:
        continue

    icd["date"] = pd.to_datetime(icd["date"], errors="coerce")
    icd = icd[icd["date"].notna()]
    icd["surgery_dt"] = icd["patient_id"].map(surgery_lookup)
    icd = icd[icd["date"] < icd["surgery_dt"]]
    if icd.empty:
        continue

    for dx_key, prefixes in DX_PATTERNS.items():
        mask = icd["code"].str.startswith(tuple(prefixes))
        for pid in icd.loc[mask, "patient_id"].unique():
            dx_flags[pid][dx_key] = 1

    if chunk_num % 50 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  ...{rows_seen:,} rows | {elapsed:.1f} min")

print(f"Done diagnosis.csv: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------
records = []
for pid in comp_ids:
    m = med_flags[pid]
    d = dx_flags[pid]
    records.append({
        "patient_id":     pid,
        "metformin":      m["metformin"],
        "rapid_insulin":  m["rapid_insulin"],
        "long_insulin":   m["long_insulin"],
        "any_insulin":    int(m["rapid_insulin"] or m["long_insulin"]),
        "glp1":           m["glp1"],
        "sglt2":          m["sglt2"],
        "dpp4":           m["dpp4"],
        "sulfonylurea":   m["sulfonylurea"],
        "tzd":            m["tzd"],
        "dm_circulatory": d["dm_circulatory"],
        "dm_other":       d["dm_other"],
        "dyslipidemia":   d["dyslipidemia"],
    })

out_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
flag_cols = [c for c in out_df.columns if c != "patient_id"]
print("\nQA — flag prevalence (comparator):")
for col in flag_cols:
    n   = out_df[col].sum()
    pct = n / len(out_df) * 100
    print(f"  {col:<20}: {n:,} ({pct:.1f}%)")

try:
    gp = pd.read_csv(GP_COVARIATES)
    gp = gp.rename(columns={"dm_circ": "dm_circulatory"})
    shared = [c for c in flag_cols if c in gp.columns]
    if shared:
        print("\nQA — GP vs Comparator prevalence:")
        print(f"  {'Variable':<20}  {'GP':>8}  {'Comparator':>12}  {'Diff':>8}")
        print("  " + "-" * 54)
        for col in shared:
            gp_pct   = gp[col].mean() * 100
            comp_pct = out_df[col].mean() * 100
            diff     = comp_pct - gp_pct
            flag     = " ⚠ large diff" if abs(diff) > 20 else ""
            print(f"  {col:<20}  {gp_pct:>7.1f}%  {comp_pct:>11.1f}%  {diff:>+7.1f}%{flag}")
except FileNotFoundError:
    print(f"\n  ({GP_COVARIATES} not found — skipping GP cross-check)")

assert out_df["patient_id"].is_unique, "Duplicate patient_ids in output"
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows, {len(out_df.columns)} cols)")
