"""
collect_psm_covariates_diagnosis.py

Collects comorbidity covariates from diagnosis.csv for PSM,
restricted to the year before surgery (matching Sadda exactly).

Covers full GP cohort (funnel_6_final_cohort.csv) and full comparator
pool (comparator_pool_ready_for_PSM.csv) in one pass - PSM has not
been applied yet at this stage.

Comorbidities collected (ICD-10-CM, within 365 days before surgery):
  - Hypertension: I10
  - CKD: N18.x
  - Heart failure: I50.x
  - CAD: I25.1x (startswith match captures subcodes)
  - Stroke: I63.x
  - Obesity: E66.x
  - Diabetic neuropathy: E10.4x, E11.4x
  - Diabetic retinopathy: E10.3x, E11.3x
  - Diabetic nephropathy: E10.2x, E11.2x
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

GP_CSV = "funnel_6_final_cohort.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
OUTPUT_CSV = "psm_covariates_comorbidities.csv"

LOOKBACK_DAYS = 365

# ICD-10-CM prefixes for each comorbidity
COMORBIDITY_PREFIXES = {
    "hypertension":          ["I10"],
    "ckd":                   ["N18"],
    "heart_failure":         ["I50"],
    "cad":                   ["I25.1"],
    "stroke":                ["I63"],
    "obesity":               ["E66"],
    "diabetic_neuropathy":   ["E10.4", "E11.4"],
    "diabetic_retinopathy":  ["E10.3", "E11.3"],
    "diabetic_nephropathy":  ["E10.2", "E11.2"],
}

print(">>> SCRIPT VERSION: collect_psm_covariates_diagnosis_v1 <<<")

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

# Initialize flags
flags = {pid: {c: False for c in COMORBIDITY_PREFIXES} for pid in all_ids}

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for comorbidities within 1yr before surgery...")

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(DIAGNOSIS_FILE,
                              usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()
        icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
        if not icd.empty:
            icd["date"] = pd.to_datetime(icd["date"], errors="coerce")
            icd = icd[icd["date"].notna()]
            if not icd.empty:
                icd["surgery_date"] = icd["patient_id"].map(surgery_lookup)
                icd["days_before"] = (icd["surgery_date"] - icd["date"]).dt.days
                # Within 1 year before surgery
                icd = icd[(icd["days_before"] >= 1) & (icd["days_before"] <= LOOKBACK_DAYS)]
                if not icd.empty:
                    for comorbidity, prefixes in COMORBIDITY_PREFIXES.items():
                        mask = icd["code"].apply(
                            lambda c: any(c.startswith(p) for p in prefixes))
                        for pid in icd.loc[mask, "patient_id"].unique():
                            flags[pid][comorbidity] = True

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        n_any = sum(1 for v in flags.values() if any(v.values()))
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{n_any:,}/{len(all_ids):,} patients with any comorbidity)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")

# Build output
records = []
for pid in sorted(all_ids):
    row = {"patient_id": pid}
    row.update(flags[pid])
    records.append(row)

out_df = pd.DataFrame(records)
out_df["group"] = out_df["patient_id"].apply(
    lambda p: "gastroparesis" if p in gp_ids else "comparator")

print("\nComorbidity prevalence:")
for c in COMORBIDITY_PREFIXES:
    n = out_df[c].sum()
    print(f"  {c}: {n:,}/{len(out_df):,} ({100*n/len(out_df):.1f}%)")

print("\nComorbidity rates by group (GP vs comparator):")
print(out_df.groupby("group")[list(COMORBIDITY_PREFIXES.keys())].mean().round(3))

assert out_df["patient_id"].is_unique
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows)")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
