"""
collect_comparator_medications_and_dx.py

Collects 12 PSM covariates for the comparator pool:
  Medications (9): metformin, any_insulin, rapid_insulin, long_insulin,
                   glp1, sglt2, dpp4, sulfonylurea, tzd
  DM complications (2): dm_circulatory, dm_other
  Comorbidity (1): dyslipidemia

Source files:
  medication_drug.csv  — brand column used for drug matching
  diagnosis.csv        — ICD-10-CM codes

Window: strictly BEFORE bariatric_date.
"""

import re
import subprocess
import time
import pandas as pd

GCS_BASE        = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MEDICATION_FILE = f"{GCS_BASE}/medication_drug.csv"
DIAGNOSIS_FILE  = f"{GCS_BASE}/diagnosis.csv"

COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
GP_COVARIATES  = "study_covariates.csv"
OUTPUT_CSV     = "psm_covariates_comparator_meds_dx.csv"

print(">>> SCRIPT VERSION: collect_comparator_medications_and_dx_v4 <<<")

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

# medication_drug.csv columns: patient_id, start_date, brand
DATE_COL = "start_date"
DRUG_COL = "brand"
print(f"Medication file: {MEDICATION_FILE}")
print(f"Date column: '{DATE_COL}' | Drug column: '{DRUG_COL}'")

# ---------------------------------------------------------------------------
# Medication patterns — brand name matching, includes combination products
# re.escape() used for safe regex
# ---------------------------------------------------------------------------
MED_PATTERNS = {
    "metformin": [
        "metformin", "glucophage", "glumetza", "fortamet", "riomet",
        "janumet",    # metformin + sitagliptin
        "kombiglyze", # metformin + saxagliptin
        "kazano",     # metformin + alogliptin
        "jentadueto", # metformin + linagliptin
        "xigduo",     # metformin + dapagliflozin
        "synjardy",   # metformin + empagliflozin
        "invokamet",  # metformin + canagliflozin
        "segluromet", # metformin + ertugliflozin
        "trijardy",   # metformin + empagliflozin + linagliptin
        "qternmet",   # metformin + dapagliflozin + saxagliptin
        "glucovance", # metformin + glyburide
        "metaglip",   # metformin + glipizide
        "avandamet",  # metformin + rosiglitazone
        "actoplus",   # metformin + pioglitazone
        # NOTE: glyxambi (empagliflozin+linagliptin) has NO metformin — not here
    ],
    "rapid_insulin": [
        "insulin lispro", "insulin aspart", "insulin glulisine",
        "humalog", "novolog", "novorapid", "apidra",
        "admelog", "lyumjev", "fiasp",
    ],
    "long_insulin": [
        "insulin glargine", "insulin detemir", "insulin degludec",
        "lantus", "basaglar", "toujeo", "semglee", "rezvoglar",
        "levemir", "tresiba",
    ],
    "glp1": [
        "semaglutide", "liraglutide", "dulaglutide", "exenatide",
        "albiglutide", "lixisenatide", "tirzepatide",
        "ozempic", "wegovy", "rybelsus",
        "victoza", "saxenda", "trulicity",
        "byetta", "bydureon", "tanzeum", "adlyxin", "mounjaro",
        "xultophy", "soliqua",
    ],
    "sglt2": [
        "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
        "jardiance", "farxiga", "forxiga", "invokana", "steglatro",
        "glyxambi",   # empagliflozin + linagliptin
        "qtern",      # dapagliflozin + saxagliptin
        "steglujan",  # ertugliflozin + sitagliptin
    ],
    "dpp4": [
        "sitagliptin", "saxagliptin", "alogliptin", "linagliptin", "vildagliptin",
        "januvia", "onglyza", "nesina", "tradjenta", "galvus",
    ],
    "sulfonylurea": [
        "glipizide", "glyburide", "glimepiride", "glibenclamide",
        "chlorpropamide", "tolbutamide", "tolazamide",
        "glucotrol", "diabeta", "micronase", "glynase", "amaryl",
    ],
    "tzd": [
        "pioglitazone", "rosiglitazone",
        "actos", "avandia",
        "duetact",   # pioglitazone + glimepiride
        "avandaryl", # rosiglitazone + glimepiride
        "oseni",     # pioglitazone + alogliptin
    ],
}

MED_REGEX = {
    key: re.compile("|".join(map(re.escape, patterns)), re.IGNORECASE)
    for key, patterns in MED_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Diagnosis patterns — dm_other excludes E10.9/E11.9 (near-universal, useless)
# ---------------------------------------------------------------------------
DX_PATTERNS = {
    "dm_circulatory": [
        "E10.5", "E11.5",
        "E10.51", "E10.52", "E10.59",
        "E11.51", "E11.52", "E11.59",
    ],
    "dm_other": [
        "E10.6", "E11.6",  # DM with other specified complications
        "E10.8", "E11.8",  # DM with unspecified complications
    ],
    "dyslipidemia": ["E78"],
}

# ---------------------------------------------------------------------------
# Initialise result dicts
# ---------------------------------------------------------------------------
med_flags = {pid: {k: 0 for k in MED_PATTERNS} for pid in comp_ids}
dx_flags  = {pid: {k: 0 for k in DX_PATTERNS}  for pid in comp_ids}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def stream_gcs_csv(path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(
            proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize
        ):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

# ===========================================================================
# PASS 1 — medication_drug.csv
# ===========================================================================
print("\n--- PASS 1: medication_drug.csv ---")
t0 = time.time()
rows_seen = chunk_num = 0

for chunk in stream_gcs_csv(
    MEDICATION_FILE,
    usecols=["patient_id", DATE_COL, DRUG_COL],
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(comp_ids)].copy()
    if chunk.empty:
        continue

    chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
    chunk["drug_lower"] = chunk[DRUG_COL].str.lower().fillna("")
    chunk["surgery_dt"] = chunk["patient_id"].map(surgery_lookup)
    chunk = chunk[chunk[DATE_COL] < chunk["surgery_dt"]]
    if chunk.empty:
        continue

    for drug_key, regex in MED_REGEX.items():
        mask = chunk["drug_lower"].str.contains(regex, na=False)
        for pid in chunk.loc[mask, "patient_id"].unique():
            med_flags[pid][drug_key] = 1

    if chunk_num % 50 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  ...{rows_seen:,} rows | {elapsed:.1f} min")

print(f"Done medication_drug.csv: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")

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
