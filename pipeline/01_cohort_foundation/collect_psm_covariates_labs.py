"""
collect_psm_covariates_labs.py

Collects baseline lab values from lab_result.csv for PSM,
restricted to the year before surgery (matching Sadda exactly).

Labs collected (LOINC codes, closest value within 365 days before surgery):
  - A1c: 4548-4, 17856-6, 4549-2 (baseline glycemic control)
  - Creatinine: 2160-0
  - eGFR: 62238-1, 50044-7
  - Total cholesterol: 2093-3
  - LDL: 2089-1

Note: Blood pressure (SBP/DBP) is stored in vitals_signs.csv, not
lab_result.csv in this TriNetX export. Collect separately using
collect_psm_covariates_vitals.py.

Covers full GP cohort and comparator pool in one pass.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE = f"{GCS_BASE}/lab_result.csv"

GP_CSV = "cohort_FINAL_analytic.csv"
COMPARATOR_CSV = "comparator_pool_raw.csv"
OUTPUT_CSV = "psm_covariates_labs.csv"

LOOKBACK_DAYS = 365

LAB_CODES = {
    "4548-4":   "a1c_baseline",
    "17856-6":  "a1c_baseline",
    "4549-2":   "a1c_baseline",
    "2160-0":   "creatinine",
    "62238-1":  "egfr",
    "50044-7":  "egfr",
    "2093-3":   "total_cholesterol",
    "2089-1":   "ldl",
}

LAB_RANGES = {
    "a1c_baseline":      (3, 20),
    "creatinine":        (0.1, 20),
    "egfr":              (1, 200),
    "total_cholesterol": (50, 500),
    "ldl":               (10, 400),
}

print(">>> SCRIPT VERSION: collect_psm_covariates_labs_v1 <<<")

gp = pd.read_csv(GP_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "bariatric_date"])
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "bariatric_date"])

all_df = pd.concat([gp, comp], ignore_index=True)
assert all_df["patient_id"].is_unique, \
    "Duplicate patient_ids found - check GP and comparator files for overlap"
surgery_lookup = dict(zip(all_df["patient_id"],
                           pd.to_datetime(all_df["bariatric_date"], errors="coerce")))
all_ids = set(all_df["patient_id"].dropna())
gp_ids = set(gp["patient_id"].dropna())
print(f"Total patients: {len(all_ids):,} ({len(gp):,} GP + {len(comp):,} comparators)")

# Store closest pre-op value per lab per patient
# {pid: {lab_name: (date, value)}}
lab_records = {pid: {} for pid in all_ids}

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning lab_result.csv for baseline labs within 1yr before surgery...")

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(LAB_FILE,
                              usecols=["patient_id", "code_system", "code",
                                       "date", "lab_result_num_val"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()

        labs = chunk[(chunk["code_system"] == "LOINC") &
                     (chunk["code"].isin(LAB_CODES.keys()))].copy()
        if not labs.empty:
            labs["date"] = pd.to_datetime(labs["date"], errors="coerce")
            labs["value"] = pd.to_numeric(labs["lab_result_num_val"], errors="coerce")
            labs = labs[labs["date"].notna() & labs["value"].notna()]
            if not labs.empty:
                labs["lab_name"] = labs["code"].map(LAB_CODES)
                labs["surgery_date"] = labs["patient_id"].map(surgery_lookup)
                labs["days_before"] = (labs["surgery_date"] - labs["date"]).dt.days
                labs = labs[(labs["days_before"] >= 1) &
                             (labs["days_before"] <= LOOKBACK_DAYS)]
                if not labs.empty:
                    for _, row in labs.iterrows():
                        pid = row["patient_id"]
                        lab = row["lab_name"]
                        lo, hi = LAB_RANGES[lab]
                        if not (lo <= row["value"] <= hi):
                            continue
                        if (lab not in lab_records[pid] or
                                row["date"] > lab_records[pid][lab][0]):
                            lab_records[pid][lab] = (row["date"], row["value"])

    if chunk_num % 100 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        n_any = sum(1 for v in lab_records.values() if v)
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{n_any:,}/{len(all_ids):,} patients with any lab)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")

# Build output
# Sorted for deterministic column order across runs
lab_names = sorted(set(LAB_CODES.values()))
records = []
for pid in sorted(all_ids):
    row = {"patient_id": pid}
    for lab in lab_names:
        if lab in lab_records[pid]:
            row[lab] = lab_records[pid][lab][1]
            row[f"{lab}_date"] = lab_records[pid][lab][0].date()
        else:
            row[lab] = None
            row[f"{lab}_date"] = None
    records.append(row)

out_df = pd.DataFrame(records)
out_df["group"] = out_df["patient_id"].apply(
    lambda p: "gastroparesis" if p in gp_ids else "comparator")

print("\nLab coverage:")
for lab in lab_names:
    n = out_df[lab].notna().sum()
    print(f"  {lab}: {n:,}/{len(out_df):,} ({100*n/len(out_df):.1f}%)")

print("\nBaseline lab means by group (GP vs comparator):")
print(out_df.groupby("group")[lab_names].mean().round(2))

assert out_df["patient_id"].is_unique
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows)")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
