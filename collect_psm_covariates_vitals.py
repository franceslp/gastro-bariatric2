"""
collect_psm_covariates_vitals.py

Collects baseline blood pressure from vitals_signs.csv for PSM.
BP is stored in vitals, not lab_result, in this TriNetX export.

Vitals collected (LOINC codes, closest value within 365 days before surgery):
  - Systolic BP: 8480-6
  - Diastolic BP: 8462-4

Covers full GP cohort and comparator pool in one pass.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
VITALS_FILE = f"{GCS_BASE}/vitals_signs.csv"

GP_CSV = "funnel_6_final_cohort.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
OUTPUT_CSV = "psm_covariates_vitals.csv"

LOOKBACK_DAYS = 365

VITAL_CODES = {
    "8480-6": "sbp",
    "8462-4": "dbp",
}

VITAL_RANGES = {
    "sbp": (60, 250),
    "dbp": (30, 150),
}

print(">>> SCRIPT VERSION: collect_psm_covariates_vitals_v1 <<<")

gp = pd.read_csv(GP_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "bariatric_date"])
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "bariatric_date"])

all_df = pd.concat([gp, comp], ignore_index=True)
assert all_df["patient_id"].is_unique, \
    "Duplicate patient_ids found - check GP and comparator files for overlap"
surgery_lookup = dict(zip(all_df["patient_id"],
                           pd.to_datetime(all_df["bariatric_date"], errors="coerce")))
n_missing_dates = sum(1 for v in surgery_lookup.values() if pd.isna(v))
if n_missing_dates > 0:
    print(f"WARNING: {n_missing_dates:,} patients have missing bariatric dates - will be skipped")
all_ids = set(all_df["patient_id"].dropna())
gp_ids = set(gp["patient_id"].dropna())
print(f"Total patients: {len(all_ids):,} ({len(gp):,} GP + {len(comp):,} comparators)")

vital_records = {pid: {} for pid in all_ids}

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning vitals_signs.csv for baseline BP within 1yr before surgery...")

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(VITALS_FILE,
                              usecols=["patient_id", "code_system", "code",
                                       "date", "value"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()

        vitals = chunk[(chunk["code_system"] == "LOINC") &
                       (chunk["code"].isin(VITAL_CODES.keys()))].copy()
        if not vitals.empty:
            vitals["date"] = pd.to_datetime(vitals["date"], errors="coerce")
            vitals["value"] = pd.to_numeric(vitals["value"], errors="coerce")
            vitals = vitals[vitals["date"].notna() & vitals["value"].notna()]
            if not vitals.empty:
                vitals["vital_name"] = vitals["code"].map(VITAL_CODES)
                vitals["surgery_date"] = vitals["patient_id"].map(surgery_lookup)
                vitals["days_before"] = (vitals["surgery_date"] - vitals["date"]).dt.days
                vitals = vitals[(vitals["days_before"] >= 1) &
                                 (vitals["days_before"] <= LOOKBACK_DAYS)]
                if not vitals.empty:
                    for _, row in vitals.iterrows():
                        pid = row["patient_id"]
                        vital = row["vital_name"]
                        lo, hi = VITAL_RANGES[vital]
                        if not (lo <= row["value"] <= hi):
                            continue
                        if (vital not in vital_records[pid] or
                                row["date"] > vital_records[pid][vital][0]):
                            vital_records[pid][vital] = (row["date"], row["value"])

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        n_any = sum(1 for v in vital_records.values() if v)
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{n_any:,}/{len(all_ids):,} patients with BP)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")

vital_names = sorted(set(VITAL_CODES.values()))
records = []
for pid in sorted(all_ids):
    row = {"patient_id": pid}
    for vital in vital_names:
        if vital in vital_records[pid]:
            row[vital] = vital_records[pid][vital][1]
            row[f"{vital}_date"] = vital_records[pid][vital][0].date()
        else:
            row[vital] = None
            row[f"{vital}_date"] = None
    records.append(row)

out_df = pd.DataFrame(records)
out_df["group"] = out_df["patient_id"].apply(
    lambda p: "gastroparesis" if p in gp_ids else "comparator")

print("\nBP coverage:")
for vital in vital_names:
    n = out_df[vital].notna().sum()
    print(f"  {vital}: {n:,}/{len(out_df):,} ({100*n/len(out_df):.1f}%)")

print("\nBaseline BP means by group:")
print(out_df.groupby("group")[vital_names].mean().round(1))

assert out_df["patient_id"].is_unique
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows)")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
