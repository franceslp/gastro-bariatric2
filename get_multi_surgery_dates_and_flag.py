"""
get_multi_surgery_dates_and_flag.py

For the 3 multi-surgery patients in the 110-patient passing cohort:
  1. Scans procedure.csv (restricted to just 3 patients - fast) to get
     ALL their bariatric surgery dates
  2. Adds a multi_surgery_flag column to the exclusion sheet
  3. Adds their second surgery date as a separate column for Dr. Sujka
"""

import subprocess
import pandas as pd

PROCEDURE_FILE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia/procedure.csv"
EXCLUSION_CSV = "exclusion_criteria_sheet.csv"
OUTPUT_CSV = "exclusion_criteria_sheet.csv"  # overwrite in place

BARIATRIC_CPT_CODES = {"43775", "43644", "43645", "43846", "43847"}
MULTI_SURGERY_IDS = {"gwj6B", "HQ_YB", "iBCND"}

print(">>> SCRIPT VERSION: get_multi_surgery_dates_and_flag_v1 <<<")

print(f"Scanning procedure.csv for all surgery dates for {len(MULTI_SURGERY_IDS)} patients...")
proc = subprocess.Popen(["gsutil", "cat", PROCEDURE_FILE], stdout=subprocess.PIPE)
all_dates = {}
rows_seen = 0
for chunk in pd.read_csv(proc.stdout, usecols=["patient_id", "code", "date"],
                          dtype=str, chunksize=500_000):
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(MULTI_SURGERY_IDS)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        hits = chunk[chunk["code"].isin(BARIATRIC_CPT_CODES)].copy()
        if not hits.empty:
            hits["date"] = pd.to_datetime(hits["date"], format="%Y%m%d", errors="coerce")
            for pid, dates in hits.groupby("patient_id")["date"]:
                all_dates.setdefault(pid, set()).update(dates.dropna().tolist())
proc.stdout.close()
proc.wait()
print(f"  scanned {rows_seen:,} rows")

print("\nAll surgery dates per multi-surgery patient:")
second_surgery_lookup = {}
all_dates_lookup = {}
for pid in sorted(MULTI_SURGERY_IDS):
    dates = sorted(all_dates.get(pid, []))
    date_strs = ", ".join(d.strftime("%Y-%m-%d") for d in dates)
    print(f"  {pid}: {date_strs}")
    all_dates_lookup[pid] = date_strs
    if len(dates) >= 2:
        second_surgery_lookup[pid] = dates[1]  # second chronological surgery

# Load exclusion sheet and add flags
excl = pd.read_csv(EXCLUSION_CSV, dtype={"patient_id": str}, low_memory=False)

excl["multi_surgery_flag"] = excl["patient_id"].isin(MULTI_SURGERY_IDS)
excl["all_bariatric_surgery_dates"] = excl["patient_id"].map(all_dates_lookup)
excl["second_bariatric_surgery_date"] = excl["patient_id"].map(second_surgery_lookup)

n_flagged = excl["multi_surgery_flag"].sum()
print(f"\nMulti-surgery flag added: {n_flagged:,} patients flagged in exclusion sheet")

excl.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote updated {OUTPUT_CSV}")
