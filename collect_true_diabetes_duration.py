"""
collect_true_diabetes_duration.py

Computes diabetes duration for PSM by scanning the full diagnosis.csv
history for every study patient (GP cohort + comparator pool).

Definition:
  Earliest recorded E10 or E11 diagnosis in the available EHR history
  diabetes_duration_days = bariatric_date - first_diabetes_date

Both GP and comparator patients are treated identically: full-history scan,
same ICD-10-CM code filter.  This eliminates the definitional mismatch of
the proxy approach (closest-before-surgery vs. first-ever date).

WORDING NOTE FOR MANUSCRIPT:
  Use "first recorded diagnosis in available EHR history" NOT "true disease
  onset."  First-coded date reflects when the diagnosis entered the EHR, not
  necessarily when the patient developed diabetes.  Reviewers will challenge
  the word "true" if it appears in the paper.

KNOWN LIMITATION - EHR COMPLETENESS BIAS:
  Patients with sparse or fragmented care histories will have later first-
  coded dates, causing underestimation of duration.  This is an inherent
  limitation of EHR-based retrospective studies and should be acknowledged
  in the methods/limitations section.

PSM MODELLING NOTE:
  Diabetes duration is right-skewed (some patients diagnosed decades before
  surgery).  Before including in PSM, consider:
    - log-transform: log1p(diabetes_duration_days)
    - winsorizing at the 99th percentile (~30-40 yrs)
  Without transformation this covariate can dominate propensity distance.
  Both diabetes_duration_log1p and diabetes_duration_winsorized_days are
  written to the output CSV for convenience - choose at analysis time.

Methods statement (copy-paste ready):
  "Diabetes duration was defined as the interval between the earliest
  recorded ICD-10-CM E10 or E11 diagnosis in the patient's available EHR
  history and the date of bariatric surgery. Because EHR records may not
  capture the full clinical history, this variable represents a lower bound
  on true diabetes duration."
"""

import subprocess
import time
import numpy as np
import pandas as pd

GCS_BASE     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

GP_CSV         = "funnel_6_final_cohort.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
OUTPUT_CSV     = "psm_covariates_true_diabetes_duration.csv"

print(">>> SCRIPT VERSION: collect_true_diabetes_duration_v3 <<<")

# ---------------------------------------------------------------------------
# Load cohorts and build lookup tables
# ---------------------------------------------------------------------------
gp   = pd.read_csv(GP_CSV,         dtype={"patient_id": str},
                   usecols=["patient_id", "bariatric_date"])
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str},
                   usecols=["patient_id", "bariatric_date"])

# FIX #1 — drop duplicate patient_ids within each file before concat.
# If either CSV has duplicates the surgery_lookup dict will silently keep
# only the last value and the final group assignment will be wrong.
gp_dupes   = gp["patient_id"].duplicated().sum()
comp_dupes = comp["patient_id"].duplicated().sum()
if gp_dupes:
    print(f"WARNING: {gp_dupes:,} duplicate patient_ids in GP file — keeping first")
    gp = gp.drop_duplicates(subset="patient_id", keep="first")
if comp_dupes:
    print(f"WARNING: {comp_dupes:,} duplicate patient_ids in comparator file — keeping first")
    comp = comp.drop_duplicates(subset="patient_id", keep="first")

# Build stable sets for group assignment BEFORE concat
# (used later so membership is unambiguous even if a patient appeared in both)
gp_ids   = set(gp["patient_id"].dropna())
comp_ids = set(comp["patient_id"].dropna())
overlap  = gp_ids & comp_ids
if overlap:
    print(f"WARNING: {len(overlap):,} patient_ids appear in both cohorts — "
          f"will be labelled 'gastroparesis' in output")

all_df = pd.concat([gp, comp], ignore_index=True).drop_duplicates(subset="patient_id")

surgery_lookup = dict(zip(
    all_df["patient_id"],
    pd.to_datetime(all_df["bariatric_date"], errors="coerce"),
))

all_ids = set(all_df["patient_id"].dropna())
print(f"Total unique patients to scan: {len(all_ids):,}")
print(f"  GP cohort:         {len(gp_ids):,}")
print(f"  Comparator pool:   {len(comp_ids):,}")

# ---------------------------------------------------------------------------
# Stream diagnosis.csv and track first-ever E10/E11 per patient
# ---------------------------------------------------------------------------
first_diabetes = {}   # patient_id -> earliest E10/E11 date seen so far

SCRIPT_START = time.time()


def stream_gcs_csv(path, usecols, chunksize=500_000):
    """Stream a GCS CSV through gsutil cat, yield pandas chunks."""
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(
            proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize
        ):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()


print("\nScanning FULL diagnosis history for first-ever E10/E11...")

rows_seen = 0
chunk_num = 0

for chunk in stream_gcs_csv(
    DIAGNOSIS_FILE,
    usecols=["patient_id", "code_system", "code", "date"],
):
    chunk_num += 1
    rows_seen += len(chunk)

    # Filter to study patients early to reduce work
    chunk = chunk[chunk["patient_id"].isin(all_ids)]
    if chunk.empty:
        continue

    chunk = chunk.copy()
    chunk["code_system"] = chunk["code_system"].str.strip()
    chunk["code"]        = chunk["code"].str.strip()

    # Keep ICD-10-CM only
    icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
    if icd.empty:
        continue

    # FIX #2 — filter to E10/E11 BEFORE the expensive to_datetime call.
    # Avoids parsing dates for every non-diabetes ICD row.
    icd = icd[icd["code"].str.startswith(("E10", "E11"))]
    if icd.empty:
        continue

    icd["date"] = pd.to_datetime(icd["date"], errors="coerce")
    icd = icd[icd["date"].notna()]
    if icd.empty:
        continue

    # Vectorised earliest-per-patient within this chunk
    chunk_min = icd.groupby("patient_id")["date"].min()

    # FIX #3 — use dict.get() with a fallback rather than checking
    # first_diabetes[pid] is None, which breaks if pid is not yet in the dict.
    for pid, min_date in chunk_min.items():
        existing = first_diabetes.get(pid)
        if existing is None or min_date < existing:
            first_diabetes[pid] = min_date

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START) / 60
        n_found = len(first_diabetes)
        print(f"  ...{rows_seen:,} rows scanned | {elapsed:.1f} min | "
              f"{n_found:,}/{len(all_ids):,} patients with diabetes date found")

elapsed_total = (time.time() - SCRIPT_START) / 60
print(f"\nDone scanning. Total: {rows_seen:,} rows in {elapsed_total:.1f} min")
print(f"Patients with at least one E10/E11 ever: {len(first_diabetes):,}")

# ---------------------------------------------------------------------------
# Build output DataFrame
# ---------------------------------------------------------------------------
records = []
for pid in all_ids:
    surg = surgery_lookup.get(pid)
    dx   = first_diabetes.get(pid)          # None if no E10/E11 ever found

    if surg is not None and dx is not None:
        duration_days = (surg - dx).days
    else:
        duration_days = None

    records.append({
        "patient_id":              pid,
        "bariatric_date":          surg,
        "first_diabetes_date":     dx,
        "diabetes_duration_days":  duration_days,
        "diabetes_duration_years": (
            round(duration_days / 365.25, 2) if duration_days is not None else None
        ),
        # FIX #4 — GP membership checked first so overlap patients go to GP
        "group": "gastroparesis" if pid in gp_ids else "comparator",
    })

out_df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# PSM-ready transformed columns
# Log-transform: handles right skew, safe because log1p(0) = 0
# Winsorized: caps extreme values at 99th percentile to prevent one long-
# duration patient from dominating propensity distance matching.
# Choose one at analysis time based on your SMD diagnostics.
# ---------------------------------------------------------------------------
out_df["diabetes_duration_log1p"] = np.log1p(
    out_df["diabetes_duration_days"].clip(lower=0)   # clip negatives to 0 before log
)

p99 = out_df["diabetes_duration_days"].quantile(0.99)
out_df["diabetes_duration_winsorized_days"] = out_df["diabetes_duration_days"].clip(upper=p99)

print(f"\nPSM transform columns added:")
print(f"  99th percentile (winsorize cap): {p99:.0f} days "
      f"({p99/365.25:.1f} yrs)")

# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------
valid = out_df["diabetes_duration_days"].dropna()

print("\nQA CHECKS:")

n_negative = (valid < 0).sum()
if n_negative:
    print(f"  WARNING: {n_negative:,} negative durations "
          f"(first diagnosis after surgery) — investigate")
else:
    print("  Negative duration check passed")

n_long = (valid > 365 * 80).sum()
if n_long:
    print(f"  WARNING: {n_long:,} patients with >80-year duration — likely bad dates")
else:
    print("  Implausibly long duration check passed (none >80 yr)")

print(f"  Mean duration:   {valid.mean():.1f} days")
print(f"  Median duration: {valid.median():.1f} days")
print(f"  Max duration:    {valid.max():.1f} days")

print(f"\nDiabetes duration by group:")
print(out_df.groupby("group")["diabetes_duration_days"].describe().round(0))

# FIX #5 — assert after building out_df, not before
assert out_df["patient_id"].is_unique, "Duplicate patient_ids in output — investigate"

out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows)")
