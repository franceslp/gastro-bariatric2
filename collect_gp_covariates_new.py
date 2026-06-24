"""
collect_gp_covariates_new.py  

Step 3 of the revised pipeline.
Recollects all Sadda PSM covariates for the NEW GP final cohort
(funnel_6_new.csv) using identical logic to bariatric_step5_covariates.py.

INPUT:  funnel_6_new.csv
OUTPUT: study_covariates_new.csv

ICD CODES: Matched exactly to Sadda et al. eTable 1 (JAMA Surgery 2026).
  Key differences from prior versions:
    stroke      = I63 only (not I63-I69) — Sadda uses cerebral infarction only
    cad         = I25.1 only (not all I25) — Sadda uses atherosclerotic heart disease
    hypertension= I10 only (not I10-I15) — Sadda uses essential hypertension only
    dm_other    = E10.6/E11.6 only (removed E10.8/E11.8) — matches Sadda exactly
    t1dm/t2dm   = separate binary flags (no dm_type categorical) — matches Sadda

DIFFERENCES FROM SADDA (document in methods):
  - Sadda used ATC codes for medications; we use RxNorm ingredient codes.
    Same drug classes captured, different coding system.
  - Sadda used E08-E13 for diabetes inclusion; we used E10/E11 only.
    Our cohort definition is stricter.
  - Sadda's TriNetX platform handled T1/T2 overlap implicitly; we keep
    both as separate binary flags (same approach, per eTable 2).

METHODOLOGY NOTES FOR MANUSCRIPT:
  - Comorbidities/DM complications: within 365 days before surgery (lookback).
  - Baseline A1c/BMI: most recent value 1-365 days before surgery.
    Same-day labs excluded (could reflect surgical admission values).
    A1c plausibility: 2-20%. BMI plausibility: 10-100 kg/m2.
  - Insulin = rapid-acting or long-acting RxNorm ingredient codes only.
  - Medication exposure = any ingredient record within 365 days before surgery.

GCS PASSES (~3 hours total):
  Pass 1: diagnosis.csv             → DM complications, comorbidities, t1/t2dm
  Pass 2: lab_result.csv            → baseline A1c, preoperative BMI
  Pass 3: medication_ingredient.csv → 9 diabetes drug classes
"""

import subprocess
import time
import os
from collections import defaultdict
import pandas as pd

print(">>> SCRIPT VERSION: collect_gp_covariates_new_v7 <<<")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET    = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAG_FILE = f"{BUCKET}/diagnosis.csv"
LAB_FILE  = f"{BUCKET}/lab_result.csv"
MED_FILE  = f"{BUCKET}/medication_ingredient.csv"

COHORT_CSV  = "funnel_6_new.csv"
OUTPUT_CSV  = "study_covariates_new.csv"

WINDOW_DAYS = 365
CHUNK_SIZE  = 500_000

# ---------------------------------------------------------------------------
# ICD-10-CM code definitions — matched exactly to Sadda eTable 1
# Codes stored WITHOUT dots (normalization applied during streaming)
# e.g. "E102" matches E10.2, E10.20, E10.21, E10.29 etc.
# ---------------------------------------------------------------------------
DX_CODES = {
    # DM complications (Sadda eTable 1)
    "dm_renal":      ["E102", "E112"],
    "dm_neuro":      ["E104", "E114"],
    "dm_circ":       ["E105", "E115"],
    "dm_opthal":     ["E103", "E113"],
    "dm_other":      ["E106", "E116"],   # E10.6/E11.6 only per Sadda
    # Diabetes type — separate binary flags per Sadda eTable 2
    "t1dm":          ["E10"],
    "t2dm":          ["E11"],
    # Comorbidities (Sadda eTable 1 — exact codes, dot-free)
    "dyslipidemia":  ["E78"],
    "ckd":           ["N18"],
    "heart_failure": ["I50"],
    "hypertension":  ["I10"],            # Sadda: I10 only
    "cad":           ["I251"],           # Sadda: I25.1 → dot-free = I251
    "stroke":        ["I63"],            # Sadda: I63 only
}

# ---------------------------------------------------------------------------
# RxNorm ingredient codes (identical to bariatric_step5_covariates.py)
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

# LOINC codes
A1C_LOINC = {"4548-4", "17856-6", "4549-2"}
BMI_LOINC = {"39156-5"}

# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------
cohort = pd.read_csv(COHORT_CSV, dtype={"patient_id": str})
cohort["bariatric_date"] = pd.to_datetime(
    cohort["bariatric_date"], errors="coerce"
)

assert cohort["patient_id"].is_unique, \
    "Duplicate patient_ids in cohort file — fix upstream"

surgery_lookup = dict(zip(cohort["patient_id"], cohort["bariatric_date"]))
assert cohort["bariatric_date"].notna().all(), \
    "Missing surgery dates in cohort — cannot compute windows"
cohort_ids = set(cohort["patient_id"].dropna())
assert len(cohort_ids) > 0, "Cohort is empty — check funnel_6_new.csv"

print(f"New GP cohort: {len(cohort_ids):,} patients")
print(f"Input:  {COHORT_CSV}")
print(f"Output: {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Helpers
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
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    return result

# ---------------------------------------------------------------------------
# Initialise result stores
# ---------------------------------------------------------------------------
dx_flags = {pid: defaultdict(int) for pid in cohort_ids}
lab_vals = {pid: {"a1c": None, "a1c_date": None,
                  "bmi": None, "bmi_date": None} for pid in cohort_ids}
med_flags = {pid: defaultdict(int) for pid in cohort_ids}

# ===========================================================================
# PASS 1 — diagnosis.csv
# ===========================================================================
print(f"\n--- Pass 1: diagnosis.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

for chunk in stream_gcs_csv(
    DIAG_FILE, usecols=["patient_id", "code_system", "code", "date"]
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)].copy()
    if chunk.empty:
        continue

    chunk["code_system"] = chunk["code_system"].str.strip()
    icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
    if icd.empty:
        continue

    icd["code"] = icd["code"].str.strip()
    # Normalize ICD format: remove dots for consistent prefix matching
    # e.g. I25.1 and I251 both match correctly after normalization
    icd["code"] = icd["code"].str.replace(".", "", regex=False)
    icd["date"] = parse_dates(icd["date"])
    icd["surgery_dt"] = icd["patient_id"].map(surgery_lookup)
    icd = icd[icd["surgery_dt"].notna()]  # guard against missing surgery dates

    # Vectorised window filter: 1-365 days before surgery
    icd["diff"] = (icd["surgery_dt"] - icd["date"]).dt.days
    icd = icd[(icd["diff"] >= 1) & (icd["diff"] <= WINDOW_DAYS)]
    if icd.empty:
        continue

    # FIX 1 — no coarse pre-filter; match directly per DX_CODES
    for dx_key, prefixes in DX_CODES.items():
        mask = icd["code"].str.startswith(tuple(prefixes))
        for pid in icd.loc[mask, "patient_id"].unique():
            dx_flags[pid][dx_key] = 1

    if chunk_num % 50 == 0:
        print(f"  ...{rows_seen:,} rows | {(time.time()-t0)/60:.1f} min")

print(f"Done: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

# ===========================================================================
# PASS 2 — lab_result.csv
# FIX 2 — uppercase code_system before comparing
# FIX 7 — groupby per patient instead of iterrows
# ===========================================================================
print(f"\n--- Pass 2: lab_result.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

for chunk in stream_gcs_csv(
    LAB_FILE,
    usecols=["patient_id", "code_system", "code", "date", "result_num"]
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)].copy()
    if chunk.empty:
        continue

    # FIX 2 — case-insensitive LOINC match
    chunk["code_system"] = chunk["code_system"].str.upper().str.strip()
    chunk = chunk[chunk["code_system"] == "LOINC"]
    if chunk.empty:
        continue

    chunk["code"] = chunk["code"].str.strip()
    chunk = chunk[chunk["code"].isin(A1C_LOINC | BMI_LOINC)].copy()
    if chunk.empty:
        continue

    chunk["date"]       = parse_dates(chunk["date"])
    chunk["result_num"] = pd.to_numeric(chunk["result_num"], errors="coerce")
    chunk["surgery_dt"] = chunk["patient_id"].map(surgery_lookup)
    chunk = chunk[chunk["surgery_dt"].notna()]  # guard against missing surgery dates
    chunk["diff"]       = (chunk["surgery_dt"] - chunk["date"]).dt.days

    # Vectorised window: 1-365 days before surgery
    chunk = chunk[(chunk["diff"] >= 1) & (chunk["diff"] <= WINDOW_DAYS)]
    chunk = chunk[chunk["result_num"].notna()]
    if chunk.empty:
        continue

    # A1c plausibility filter
    a1c_rows = chunk[chunk["code"].isin(A1C_LOINC)].copy()
    a1c_rows = a1c_rows[
        (a1c_rows["result_num"] >= 2) & (a1c_rows["result_num"] <= 20)
    ]

    # BMI plausibility filter
    bmi_rows = chunk[chunk["code"].isin(BMI_LOINC)].copy()
    bmi_rows = bmi_rows[
        (bmi_rows["result_num"] >= 10) & (bmi_rows["result_num"] <= 100)
    ]

    # FIX 4 — sort by date descending, take most recent calendar date
    # (most recent within window = most representative of pre-op status)
    for pid, grp in a1c_rows.groupby("patient_id"):
        best = grp.sort_values("date", ascending=False).iloc[0]
        existing_dt = lab_vals[pid]["a1c_date"]
        if existing_dt is None or best["date"] > existing_dt:
            lab_vals[pid]["a1c"]      = best["result_num"]
            lab_vals[pid]["a1c_date"] = best["date"]

    for pid, grp in bmi_rows.groupby("patient_id"):
        best = grp.sort_values("date", ascending=False).iloc[0]
        existing_dt = lab_vals[pid]["bmi_date"]
        if existing_dt is None or best["date"] > existing_dt:
            lab_vals[pid]["bmi"]      = best["result_num"]
            lab_vals[pid]["bmi_date"] = best["date"]

    if chunk_num % 50 == 0:
        print(f"  ...{rows_seen:,} rows | {(time.time()-t0)/60:.1f} min")

print(f"Done: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

# ===========================================================================
# PASS 3 — medication_ingredient.csv
# FIX 3 — vectorised window, no apply/lambda
# ===========================================================================
print(f"\n--- Pass 3: medication_ingredient.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

for chunk in stream_gcs_csv(
    MED_FILE, usecols=["patient_id", "code", "start_date"]
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)].copy()
    if chunk.empty:
        continue

    # Normalize code strings before matching (handles whitespace, type inconsistency)
    chunk["code"] = chunk["code"].astype(str).str.strip()
    chunk = chunk[chunk["code"].isin(code_to_class)]
    if chunk.empty:
        continue

    chunk["start_date"] = parse_dates(chunk["start_date"])
    chunk["surgery_dt"] = chunk["patient_id"].map(surgery_lookup)
    chunk = chunk[chunk["surgery_dt"].notna()]  # guard against missing surgery dates

    # FIX 3 — vectorised window filter
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

# ---------------------------------------------------------------------------
# Build output
# FIX 6 — Issue 6: add dm_type with T2DM-precedence hierarchy
# FIX 5 — any_insulin uses integer arithmetic
# ---------------------------------------------------------------------------
records = []
for pid in cohort_ids:
    dx  = dx_flags[pid]
    lab = lab_vals[pid]
    med = med_flags[pid]

    # T1DM and T2DM kept as separate binary flags — matches Sadda eTable 2 exactly.
    # Dual-coded patients (both E10+E11) will have both flags=1.
    # This is expected in EHR data and Sadda did not resolve overlap.
    records.append({
        "patient_id":           pid,
        "bariatric_date":       surgery_lookup[pid],
        # Labs
        "baseline_a1c":         lab["a1c"],
        "baseline_a1c_date":    lab["a1c_date"],
        "baseline_a1c_missing": int(lab["a1c"] is None),
        "preoperative_bmi":         lab["bmi"],
        "preoperative_bmi_missing": int(lab["bmi"] is None),
        # DM type — separate binary flags (no categorical, per Sadda eTable 2)
        "t1dm":          dx["t1dm"],
        "t2dm":          dx["t2dm"],
        # DM complications
        "dm_renal":      dx["dm_renal"],
        "dm_neuro":      dx["dm_neuro"],
        "dm_circ":       dx["dm_circ"],
        "dm_opthal":     dx["dm_opthal"],
        "dm_other":      dx["dm_other"],
        # Comorbidities
        "dyslipidemia":  dx["dyslipidemia"],
        "hypertension":  dx["hypertension"],
        "ckd":           dx["ckd"],
        "heart_failure": dx["heart_failure"],
        "cad":           dx["cad"],
        "stroke":        dx["stroke"],
        # Medications
        "metformin":     med["metformin"],
        "rapid_insulin": med["rapid_insulin"],
        "long_insulin":  med["long_insulin"],
        # FIX 5 — integer arithmetic not bool
        "any_insulin":   int((med["rapid_insulin"] + med["long_insulin"]) > 0),
        "glp1":          med["glp1"],
        "sglt2":         med["sglt2"],
        "dpp4":          med["dpp4"],
        "sulfonylurea":  med["sulfonylurea"],
        "tzd":           med["tzd"],
    })

out_df = pd.DataFrame(records)

# FIX 4 — merge demographics from cohort file (safer than per-row .loc)
demo_cols = [c for c in ["sex", "race", "ethnicity",
                          "surgery_type", "age_at_surgery_approx"]
             if c in cohort.columns]
cohort_lookup = cohort[["patient_id"] + demo_cols].copy()
assert cohort_lookup["patient_id"].is_unique, \
    "Duplicate patient_ids in cohort_lookup — merge would multiply rows"
cohort_lookup = cohort_lookup.rename(columns={
    "surgery_type":          "procedure_type",
    "age_at_surgery_approx": "age_at_surgery",
})
out_df = out_df.merge(cohort_lookup, on="patient_id", how="left")

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
assert len(out_df) == len(cohort_ids), \
    f"Lost patients: expected {len(cohort_ids)}, got {len(out_df)}"
assert out_df["patient_id"].nunique() == len(out_df), \
    "Duplicate patient_ids in output"
assert out_df["bariatric_date"].notna().all(), \
    "Missing bariatric_date for some patients"

t1t2_overlap = ((out_df["t1dm"] == 1) & (out_df["t2dm"] == 1)).sum()
if t1t2_overlap > 0:
    print(f"NOTE: {t1t2_overlap} patients have both E10+E11 coded. "
          f"Both flags kept as 1 (matches Sadda approach). "
          f"Sadda performed T1/T2 subgroup analyses separately.")
else:
    print("T1DM/T2DM overlap: none ✓")

print("All assertions passed ✓")
print("NOTE: Medication exposure = any ingredient record 1-365 days before "
      "surgery. Active/inactive not distinguished. Document in methods.")

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
# Bug fix — flag_cols explicitly defined (missing = NameError crash)
flag_cols = [
    "t1dm", "t2dm",
    "dm_renal", "dm_neuro", "dm_circ", "dm_opthal", "dm_other",
    "dyslipidemia", "hypertension", "ckd", "heart_failure", "cad", "stroke",
    "metformin", "rapid_insulin", "long_insulin", "any_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
]

print(f"\nQA — {OUTPUT_CSV}:")
print(f"  Rows: {len(out_df):,} | Cols: {len(out_df.columns)}")

print(f"\n  A1c missingness:  {out_df['baseline_a1c'].isna().sum()} "
      f"({out_df['baseline_a1c'].isna().mean()*100:.1f}%)")
print(f"  BMI missingness:  {out_df['preoperative_bmi'].isna().sum()} "
      f"({out_df['preoperative_bmi'].isna().mean()*100:.1f}%)")

if out_df["baseline_a1c"].notna().any():
    print(f"  A1c range: {out_df['baseline_a1c'].min():.1f} – "
          f"{out_df['baseline_a1c'].max():.1f}")
if out_df["preoperative_bmi"].notna().any():
    print(f"  BMI range: {out_df['preoperative_bmi'].min():.1f} – "
          f"{out_df['preoperative_bmi'].max():.1f}")

print(f"\n  Flag prevalence (t1dm/t2dm are separate binary flags per Sadda):")
for col in flag_cols:
    if col in out_df.columns:
        n_flag = out_df[col].sum()
        print(f"    {col:<20}: {n_flag:3d} ({n_flag/len(out_df)*100:.1f}%)")

out_df.to_csv(OUTPUT_CSV, index=False)
file_size_kb = os.path.getsize(OUTPUT_CSV) / 1024
print(f"\nWrote {OUTPUT_CSV}")
print(f"  Rows: {len(out_df):,} | Cols: {len(out_df.columns)} | "
      f"Size: {file_size_kb:.1f} KB")
print("Done.")
