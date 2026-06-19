"""
check_E10_E11_prior_to_K3184_bariatric_subset.py

For the bariatric-surgery subset of the cohort, checks whether E10 (Type 1)
or E11 (Type 2) diabetes codes SPECIFICALLY appear within 1 year before the
patient's first K31.84 diagnosis date.

WHY THIS NEEDS A NEW SCAN: the existing has_type1_diabetes_code /
has_type2_diabetes_code columns are lifetime "ever has this code" flags
with no date attached - they can't tell you whether the E10/E11 code came
before or after K31.84. And first_diabetes_dx_date is the earliest date
across the WHOLE E08-E13 block combined, not specifically E10 or E11 - so
if a patient's earliest diabetes code was, say, E13 (other specified) but
their E11 code came later, first_diabetes_dx_date wouldn't tell you when
the E11 specifically appeared. This script captures first_E10_date and
first_E11_date separately, which is what's actually needed here.

WINDOW: 1 year (365 days) before K31.84 dx, SAME-DAY INCLUDED. Same-day
inclusion mirrors the earlier GES-to-diagnosis decision elsewhere in this
pipeline - a diabetes diagnosis coded on the same encounter/day as a new
gastroparesis diagnosis is clinically plausible, so it isn't excluded.

CRITICAL FIX (caught in review, see chat): the windowed flag is NOT computed
from first_E10_date/first_E11_date. Diabetes is a chronic diagnosis that
gets re-coded at essentially every visit once present, so a patient's
FIRST-EVER E10/E11 code can easily be years before K31.84 even when they
ALSO have a recent occurrence inside the 1-year window. Anchoring on the
first-ever date alone would silently misclassify those patients as not
meeting the window. Instead, EVERY E10/E11 row is checked individually
against that patient's own K31.84 date during the scan, and the patient is
flagged in-window if ANY occurrence (not just the first) falls inside
[dx, dx-365]. The unbounded "any time before" flag is unaffected by this -
first-date logic IS valid there, since proving the earliest-ever occurrence
predates dx is sufficient on its own (there's no upper bound to miss
occurrences inside).

DEFENSIVE FIXES applied (none were confirmed live bugs, but cheap insurance):
  - has_bariatric_surgery is explicitly coerced to real bool, in case the
    CSV was ever hand-edited or re-saved in a way that altered its dtype.
    (Reading with dtype={"patient_id": str} only, as this script does,
    should already get pandas' automatic True/False string inference
    correct for any other column - but this removes any doubt.)
  - .str.startswith(..., na=False) added so rows with a missing/NaN code
    value can't silently produce NaN in a boolean mask.

SCOPE: restricted to the bariatric-surgery subset only (patients with
has_bariatric_surgery == True), not the full 263,087-patient cohort - this
question only concerns that subset, so there's no reason to scan more
broadly. This keeps the scan fast despite touching diagnosis.csv again.

NOTE ON AMBIGUOUS TYPE: a patient can have both E10 and E11 within the
window. This is flagged, not resolved - in real EHR data this is usually
miscoding, type reclassification, or coding drift over a long record, not
necessarily evidence the patient genuinely had both types. Needs a manual
call, not a script-level fix.

Run this on the bariatric-and-concurrency output file. Writes a NEW file.
"""

import os
import subprocess
import time
import pandas as pd
from pandas.api.types import is_bool_dtype

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "bariatric_subset_with_E10_E11_timing.csv"

E10_PREFIX = "E10"
E11_PREFIX = "E11"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Missing required input file: {INPUT_CSV}")

print(f"Loading {INPUT_CSV}...")
full_df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = ["has_bariatric_surgery", "first_K31_84_date", "in_study_period"]
missing_cols = [c for c in required_cols if c not in full_df.columns]
if missing_cols:
    raise ValueError(f"{INPUT_CSV} is missing expected columns: {missing_cols}")

# Defensive: coerce to real bool in case the CSV was ever hand-edited or
# re-saved in a way that altered this column's dtype (e.g. "True"/"False"
# as uppercase, or 1/0). Cheap insurance, not a confirmed bug.
if not is_bool_dtype(full_df["has_bariatric_surgery"]):
    full_df["has_bariatric_surgery"] = (
        full_df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")
    )

# Restrict to the bariatric-surgery subset only - that's the population this
# question is actually about.
bariatric_df = full_df[full_df["has_bariatric_surgery"]].copy()
bariatric_ids = set(bariatric_df["patient_id"])
print(f"  bariatric-surgery subset: {len(bariatric_ids):,} patients")
print(f"  of those, in-study-period K31.84: {(bariatric_df['in_study_period']).sum():,}")

dx_date_dt = pd.to_datetime(bariatric_df["first_K31_84_date"], errors="coerce")
dx_date_lookup = dict(zip(bariatric_df["patient_id"], dx_date_dt))

WINDOW_DAYS = 365  # 1 year

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for E10/E11 codes, restricted to the {len(bariatric_ids):,}-patient bariatric subset...")

# Track BOTH the earliest-ever date (valid for the "any time before"
# descriptive flag - first-date logic is fine there, since if the first-
# ever occurrence predates dx, that alone proves evidence exists before dx)
# AND whether ANY occurrence (not just the first) falls within the 1-year
# window. The window check needs every row, not just the minimum date,
# since a chronic diagnosis like diabetes gets re-coded at every visit -
# the patient's FIRST-EVER E10/E11 code could be years before K31.84 even
# while they also have a recent one right before their gastroparesis dx.
first_E10_date = {}
first_E11_date = {}
E10_in_window_patients = set()
E11_in_window_patients = set()

rows_seen = 0
chunk_num = 0
diag_start = time.time()
for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(bariatric_ids)]
    if not chunk.empty:
        icd_chunk = chunk[chunk["code_system"] == "ICD-10-CM"]
        if not icd_chunk.empty:
            icd_chunk = icd_chunk.copy()
            icd_chunk["date"] = pd.to_datetime(icd_chunk["date"], format="%Y%m%d", errors="coerce")
            icd_chunk["dx_date"] = icd_chunk["patient_id"].map(dx_date_lookup)
            days_dx_after_row = (icd_chunk["dx_date"] - icd_chunk["date"]).dt.days
            in_window_mask = (
                icd_chunk["dx_date"].notna()
                & icd_chunk["date"].notna()
                & (days_dx_after_row >= 0)
                & (days_dx_after_row <= WINDOW_DAYS)
            )

            e10_hits = icd_chunk[icd_chunk["code"].str.startswith(E10_PREFIX, na=False)]
            if not e10_hits.empty:
                e10_min = e10_hits.groupby("patient_id")["date"].min().to_dict()
                for pid, d in e10_min.items():
                    if pd.notna(d) and (pid not in first_E10_date or d < first_E10_date[pid]):
                        first_E10_date[pid] = d
                e10_window_hits = e10_hits[in_window_mask.loc[e10_hits.index]]
                if not e10_window_hits.empty:
                    E10_in_window_patients.update(e10_window_hits["patient_id"])

            e11_hits = icd_chunk[icd_chunk["code"].str.startswith(E11_PREFIX, na=False)]
            if not e11_hits.empty:
                e11_min = e11_hits.groupby("patient_id")["date"].min().to_dict()
                for pid, d in e11_min.items():
                    if pd.notna(d) and (pid not in first_E11_date or d < first_E11_date[pid]):
                        first_E11_date[pid] = d
                e11_window_hits = e11_hits[in_window_mask.loc[e11_hits.index]]
                if not e11_window_hits.empty:
                    E11_in_window_patients.update(e11_window_hits["patient_id"])

    if chunk_num % 50 == 0:
        elapsed_min = (time.time() - diag_start) / 60
        print(f"    ...diagnosis.csv: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed)")

print(f"\n  scanned {rows_seen:,} diagnosis rows")
print(f"  bariatric-subset patients with an E10 code (any time): {len(first_E10_date):,}")
print(f"  bariatric-subset patients with an E11 code (any time): {len(first_E11_date):,}")

bariatric_df["first_E10_date"] = bariatric_df["patient_id"].map(first_E10_date)
bariatric_df["first_E11_date"] = bariatric_df["patient_id"].map(first_E11_date)

first_E10_dt = pd.to_datetime(bariatric_df["first_E10_date"], errors="coerce")
first_E11_dt = pd.to_datetime(bariatric_df["first_E11_date"], errors="coerce")
dx_dt = pd.to_datetime(bariatric_df["first_K31_84_date"], errors="coerce")

# "Any time before" - first-date logic IS valid here: if the earliest-ever
# occurrence predates dx, that alone proves evidence exists before dx (no
# window to miss occurrences inside, since "any time" has no upper bound).
days_dx_after_first_E10 = (dx_dt - first_E10_dt).dt.days
days_dx_after_first_E11 = (dx_dt - first_E11_dt).dt.days
bariatric_df["E10_before_K31_84_any_time"] = first_E10_dt.notna() & dx_dt.notna() & (days_dx_after_first_E10 >= 0)
bariatric_df["E11_before_K31_84_any_time"] = first_E11_dt.notna() & dx_dt.notna() & (days_dx_after_first_E11 >= 0)

# "Within 1 year before" - uses the per-row window membership computed
# DURING the scan above (E10_in_window_patients / E11_in_window_patients),
# NOT first-date arithmetic. This is the fix for the bug where relying on
# first_E10_date/first_E11_date alone would miss patients whose first-ever
# occurrence was years before dx but who ALSO have a more recent occurrence
# inside the 1-year window - exactly the scenario a chronic, repeatedly-
# coded diagnosis like diabetes produces routinely.
bariatric_df["E10_within_1yr_before_K31_84"] = bariatric_df["patient_id"].isin(E10_in_window_patients)
bariatric_df["E11_within_1yr_before_K31_84"] = bariatric_df["patient_id"].isin(E11_in_window_patients)
bariatric_df["E10_or_E11_within_1yr_before_K31_84"] = (
    bariatric_df["E10_within_1yr_before_K31_84"] | bariatric_df["E11_within_1yr_before_K31_84"]
)

print("\nSANITY CHECK:")
in_period = bariatric_df["in_study_period"]
n_total = in_period.sum()
print(f"\nOf the {n_total:,} in-study-period K31.84 bariatric-surgery patients:")
print(f"  E10 (Type 1) any time before K31.84 dx:        {(in_period & bariatric_df['E10_before_K31_84_any_time']).sum():,}")
print(f"  E11 (Type 2) any time before K31.84 dx:        {(in_period & bariatric_df['E11_before_K31_84_any_time']).sum():,}")
print(f"  E10 (Type 1) within 1yr before K31.84 dx:      {(in_period & bariatric_df['E10_within_1yr_before_K31_84']).sum():,}")
print(f"  E11 (Type 2) within 1yr before K31.84 dx:      {(in_period & bariatric_df['E11_within_1yr_before_K31_84']).sum():,}")
print(f"  E10 OR E11 within 1yr before K31.84 dx:        {(in_period & bariatric_df['E10_or_E11_within_1yr_before_K31_84']).sum():,}")
both_window = in_period & bariatric_df["E10_within_1yr_before_K31_84"] & bariatric_df["E11_within_1yr_before_K31_84"]
print(f"  (both E10 AND E11 within 1yr before dx - ambiguous type: {both_window.sum():,})")

bariatric_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(bariatric_df):,} rows) with first_E10_date, first_E11_date,")
print("E10_before_K31_84_any_time, E11_before_K31_84_any_time,")
print("E10_within_1yr_before_K31_84, E11_within_1yr_before_K31_84, and")
print("E10_or_E11_within_1yr_before_K31_84 columns.")

total_elapsed_min = (time.time() - SCRIPT_START_TIME) / 60
print(f"\nTotal script runtime: {total_elapsed_min:.1f} minutes")
