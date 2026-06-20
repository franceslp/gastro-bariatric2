"""
find_K3184_in_window_real_scan.py

REPLACES the flawed two-separate-checks approach (Step 1 using last_K31_84_date,
Step 3 using first_K31_84_date) with ONE precise question, checked against
EVERY individual K31.84 occurrence per patient, not just first/last:

  Does this patient have at least one K31.84 occurrence strictly between
  Oct 1, 2015 and their surgery date?

This single check replaces what Steps 1 + 3 were both trying to capture
together. Surgery being post-Oct-2015 falls out automatically as a
consequence (a qualifying K31.84 code must be >= Oct 2015, and surgery must
be after that code, so surgery is necessarily also post-Oct-2015).

Restricted to the 2,842 bariatric-surgery patients, since the window is
defined relative to each patient's own surgery date.
"""

import subprocess
import time
import pandas as pd
from pandas.api.types import is_bool_dtype

print(">>> SCRIPT VERSION: find_K3184_in_window_real_scan_v1 <<<")

STUDY_START = pd.Timestamp("2015-10-01")
GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "bariatric_patients_K3184_window_check.csv"

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)
if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

df = df[df["has_bariatric_surgery"]].copy()
cohort_ids = set(df["patient_id"])
print(f"Bariatric-surgery patients: {len(cohort_ids):,}")

# Flag (not duplicate) patients with multiple surgery dates - bariatric_date
# is already a single value per patient (one row per patient in this file),
# so this can't cause row duplication - just surfacing it for later review,
# since "which surgery" matters for someone with more than one.
if "num_distinct_bariatric_surgery_dates" in df.columns:
    multi_surgery = df["num_distinct_bariatric_surgery_dates"] > 1
    print(f"Patients with MULTIPLE distinct surgery dates (flagged for later review, not duplicated): {multi_surgery.sum():,}")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")

# Transparency counts - missing/invalid surgery dates can't be evaluated at
# all (no date to compare against), and a pre-Oct-2015 surgery date would be
# a data-quality anomaly worth knowing about, even though the filter logic
# below is mathematically guaranteed to never count such a patient as
# qualifying regardless (Oct2015 <= date < surgery_date < Oct2015 is a
# direct contradiction, so this can't silently bias the main count).
n_missing_surgery_date = surgery_dt.isna().sum()
n_pre2015_surgery = (surgery_dt.notna() & (surgery_dt < STUDY_START)).sum()
print(f"Missing/unparseable surgery dates: {n_missing_surgery_date:,}")
print(f"Surgery dates themselves before Oct 2015 (data-quality flag, same anomaly seen with K31.84 dates earlier): {n_pre2015_surgery:,}")

surgery_lookup = dict(zip(df["patient_id"], surgery_dt))

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for K31.84 occurrences strictly within (Oct 1 2015, surgery_date),")
print(f"restricted to {len(cohort_ids):,} patients...")

has_qualifying_occurrence = set()
qualifying_dates = {}
rows_seen = 0
chunk_num = 0

for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        k31 = chunk[(chunk["code_system"] == "ICD-10-CM") & (chunk["code"] == "K31.84")].copy()
        if not k31.empty:
            k31["date"] = pd.to_datetime(k31["date"], format="%Y%m%d", errors="coerce")
            k31["surgery_date"] = k31["patient_id"].map(surgery_lookup)
            in_window = k31["date"].notna() & k31["surgery_date"].notna() & (k31["date"] >= STUDY_START) & (k31["date"] < k31["surgery_date"])
            qualifying = k31[in_window]
            if not qualifying.empty:
                has_qualifying_occurrence.update(qualifying["patient_id"])
                # track the LATEST qualifying occurrence per patient (closest to surgery, still within window)
                for pid, d in qualifying.groupby("patient_id")["date"].max().items():
                    if pid not in qualifying_dates or d > qualifying_dates[pid]:
                        qualifying_dates[pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, {len(has_qualifying_occurrence):,} qualifying patients found so far)")

print(f"\n  done - scanned {rows_seen:,} rows")

df["has_K3184_strictly_in_window"] = df["patient_id"].isin(has_qualifying_occurrence)
df["latest_K3184_in_window"] = df["patient_id"].map(qualifying_dates)

n_qualifying = df["has_K3184_strictly_in_window"].sum()
print(f"\nOf {len(df):,} bariatric-surgery patients:")
print(f"  Have a K31.84 occurrence strictly within (Oct 1 2015, surgery_date): {n_qualifying:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
