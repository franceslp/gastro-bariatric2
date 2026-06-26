"""
collect_comparator_medications_and_dxNEW.py  

Collects 12 PSM covariates for the comparator pool using methodology
IDENTICAL to collect_gp_covariates_new.py:

  Source: medication_ingredient.csv + RxNorm ingredient codes
  Window: 1-365 days before bariatric_date (matches GP cohort exactly)
           Same-day records excluded (consistent with GP pipeline)

  Medications (9): metformin, any_insulin, rapid_insulin, long_insulin,
                   glp1, sglt2, dpp4, sulfonylurea, tzd
  DM complications (2): dm_circulatory, dm_other
  Comorbidity (1): dyslipidemia
  Diabetes type (2): t1dm, t2dm (binary flags, same as GP)

ICD codes mapped to Sadda et al. eTable 1 diabetes complication categories.

KEY FIXES vs v6:
  - Window changed from lifetime pre-op to 1-365 days (matches GP)
  - Removed global ICD prefix pre-filter (matches GP exactly)
  - any_insulin uses integer arithmetic (matches GP)
  - Added t1dm/t2dm flags (symmetric with GP pipeline)
  - Added t1/t2 overlap QA
"""

import subprocess
import time
import os
import pandas as pd
from collections import defaultdict

GCS_BASE        = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MEDICATION_FILE = f"{GCS_BASE}/medication_ingredient.csv"
DIAGNOSIS_FILE  = f"{GCS_BASE}/diagnosis.csv"

COMPARATOR_CSV = "comparator_pool_raw.csv"
GP_COVARIATES  = "study_covariates_new.csv"   # compare against new GP covariates
OUTPUT_CSV     = "psm_covariates_comparator_meds_dx.csv"

WINDOW_DAYS = 365   # matches GP cohort exactly
CHUNK_SIZE  = 1_000_000

print(">>> SCRIPT VERSION: collect_comparator_medications_and_dx_v8 <<<")

# ---------------------------------------------------------------------------
# Load comparator pool
# ---------------------------------------------------------------------------
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str},
                   usecols=["patient_id", "bariatric_date"])
comp = comp.drop_duplicates(subset="patient_id")
comp["surgery_dt"] = pd.to_datetime(comp["bariatric_date"], errors="coerce")

assert comp["surgery_dt"].notna().all(), \
    "Missing surgery dates in comparator — cannot compute windows"

surgery_lookup = dict(zip(comp["patient_id"], comp["surgery_dt"]))
comp_ids = set(comp["patient_id"].dropna())
print(f"Comparator patients: {len(comp_ids):,}")
print(f"Window: 1-{WINDOW_DAYS} days before surgery (matches GP cohort)")

# ---------------------------------------------------------------------------
# RxNorm ingredient codes — IDENTICAL to GP cohort script
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
code_to_class = {
    code: cls for cls, codes in MED_CODES.items() for code in codes
}

# ---------------------------------------------------------------------------
# ICD-10-CM code definitions — mapped to Sadda et al. eTable 1 categories
# Dot-free format for consistent prefix matching after normalization
# ---------------------------------------------------------------------------
DX_CODES = {
    "dm_circulatory": ["E105", "E115"],   # E10.5/E11.5 dot-free
    "dm_other":       ["E106", "E116"],   # E10.6/E11.6 only per Sadda
    "dyslipidemia":   ["E78"],
    "t1dm":           ["E10"],
    "t2dm":           ["E11"],
}

# ---------------------------------------------------------------------------
# Initialise result dicts
# ---------------------------------------------------------------------------
med_flags = {pid: defaultdict(int) for pid in comp_ids}
dx_flags  = {pid: {k: 0 for k in DX_CODES} for pid in comp_ids}

# ---------------------------------------------------------------------------
# Helpers — identical to GP script
# ---------------------------------------------------------------------------
def stream_gcs_csv(path, usecols, chunksize=CHUNK_SIZE):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(
            proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize
        ):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

def parse_dates(series):
    """Handle both YYYYMMDD and ISO date formats — identical to GP script."""
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    return result

# ===========================================================================
# PASS 1 — medication_ingredient.csv
# Window: 1-365 days before surgery (FIX: was lifetime pre-op)
# ===========================================================================
print("\n--- PASS 1: medication_ingredient.csv ---")
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

    # Normalize code strings before matching
    chunk["code"] = chunk["code"].astype(str).str.strip()
    chunk = chunk[chunk["code"].isin(code_to_class)]
    if chunk.empty:
        continue

    chunk["start_date"] = parse_dates(chunk["start_date"])
    chunk["surgery_dt"] = chunk["patient_id"].map(surgery_lookup)

    # Guard against missing surgery dates before computing diff
    chunk = chunk[chunk["surgery_dt"].notna()]

    # FIX 2 — vectorised 1-365 day window (matches GP exactly)
    chunk["diff"] = (chunk["surgery_dt"] - chunk["start_date"]).dt.days
    chunk = chunk[(chunk["diff"] >= 1) & (chunk["diff"] <= WINDOW_DAYS)]
    if chunk.empty:
        continue

    chunk["drug_class"] = chunk["code"].map(code_to_class)
    for pid, drug_class in zip(chunk["patient_id"], chunk["drug_class"]):
        med_flags[pid][drug_class] = 1

    if chunk_num % 50 == 0:
        print(f"  ...{rows_seen:,} rows | {(time.time()-t0)/60:.1f} min")

print(f"Done: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

# ===========================================================================
# PASS 2 — diagnosis.csv
# FIX 3 — no global prefix pre-filter (matches GP pipeline exactly)
# ===========================================================================
print("\n--- PASS 2: diagnosis.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

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
    # Normalize ICD format: remove dots for consistent prefix matching
    icd["code"] = icd["code"].str.replace(".", "", regex=False)
    icd["date"] = pd.to_datetime(icd["date"], errors="coerce")
    icd = icd[icd["date"].notna()]
    icd["surgery_dt"] = icd["patient_id"].map(surgery_lookup)

    # Guard against missing surgery dates before computing diff
    icd = icd[icd["surgery_dt"].notna()]

    # Vectorised 1-365 day window
    icd["diff"] = (icd["surgery_dt"] - icd["date"]).dt.days
    icd = icd[(icd["diff"] >= 1) & (icd["diff"] <= WINDOW_DAYS)]
    if icd.empty:
        continue

    # No global pre-filter — match directly per DX_CODES (matches GP)
    for dx_key, prefixes in DX_CODES.items():
        mask = icd["code"].str.startswith(tuple(prefixes))
        for pid in icd.loc[mask, "patient_id"].unique():
            dx_flags[pid][dx_key] = 1

    if chunk_num % 50 == 0:
        print(f"  ...{rows_seen:,} rows | {(time.time()-t0)/60:.1f} min")

print(f"Done: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

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
        # FIX 1 — integer arithmetic matches GP pipeline exactly
        "any_insulin":    int((m["rapid_insulin"] + m["long_insulin"]) > 0),
        "glp1":           m["glp1"],
        "sglt2":          m["sglt2"],
        "dpp4":           m["dpp4"],
        "sulfonylurea":   m["sulfonylurea"],
        "tzd":            m["tzd"],
        "dm_circulatory": d["dm_circulatory"],
        "dm_other":       d["dm_other"],
        "dyslipidemia":   d["dyslipidemia"],
        # FIX 5 — t1dm/t2dm flags symmetric with GP pipeline
        "t1dm":           d["t1dm"],
        "t2dm":           d["t2dm"],
    })

out_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
assert out_df["patient_id"].is_unique, "Duplicate patient_ids in output"

# Bug 1 fix — flag_cols explicitly defined (was missing, would crash)
flag_cols = [
    "t1dm", "t2dm",
    "dm_circulatory", "dm_other", "dyslipidemia",
    "metformin", "rapid_insulin", "long_insulin", "any_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
]

# T1/T2 overlap check
t1t2_overlap = ((out_df["t1dm"] == 1) & (out_df["t2dm"] == 1)).sum()
print(f"\nT1DM+T2DM overlap (both=1): {t1t2_overlap:,} "
      f"({'expected — kept as separate flags per Sadda' if t1t2_overlap > 0 else '✓ none'})")

flag_cols = [c for c in out_df.columns if c != "patient_id"]
print("\nQA — flag prevalence (comparator):")
for col in flag_cols:
    n   = out_df[col].sum()
    pct = n / len(out_df) * 100
    print(f"  {col:<20}: {n:,} ({pct:.1f}%)")

# Cross-check vs GP cohort
try:
    gp = pd.read_csv(GP_COVARIATES)
    # align column names
    gp = gp.rename(columns={"dm_circ": "dm_circulatory"})
    shared = [c for c in flag_cols if c in gp.columns]
    if shared:
        print("\nQA — GP vs Comparator prevalence (same 365-day window):")
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

print("\nMissingness check (all binary flags should be 0 — no imputation):")
print(out_df.isna().sum().sort_values(ascending=False).head(10).to_string())

out_df.to_csv(OUTPUT_CSV, index=False)
file_size_kb = os.path.getsize(OUTPUT_CSV) / 1024
print(f"\nWrote {OUTPUT_CSV}")
print(f"  Rows: {len(out_df):,} | Cols: {len(out_df.columns)} | Size: {file_size_kb:.1f} KB")
