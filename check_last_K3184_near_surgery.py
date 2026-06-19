"""
check_last_K3184_near_surgery.py

SIMPLE QUESTION: for the 1,126 patients who qualify under the 5yr E10/E11
rule, what's their MOST RECENT (not first) K31.84 diagnosis date, and how
close is that to their surgery date?

WHY THIS MATTERS: a patient's first K31.84 code just shows they were
diagnosed with gastroparesis at SOME point - it doesn't mean the condition
was still present by the time of surgery (gastroparesis can improve with
treatment). If a patient ALSO has a K31.84 code close to surgery, that's
better evidence the diagnosis was still active/being managed near surgery,
not just a one-time historical diagnosis.

This needs a fresh scan - the existing files only ever saved the FIRST
K31.84 date, never the most recent one.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "bariatric_subset_5yr_concurrency_E10_E11.csv"
OUTPUT_CSV = "qualified_patients_with_last_K3184_date.csv"

print(">>> SCRIPT VERSION: last_K3184_near_surgery_v1 <<<")  # marker to confirm the right version ran

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

qualified = df[(df["meets_5yr_rule_E10_E11"] == True) & (df["in_study_period"] == True)].copy()
qualified_ids = set(qualified["patient_id"])
print(f"Qualifying patients: {len(qualified_ids):,}\n")

surgery_lookup = dict(zip(qualified["patient_id"], pd.to_datetime(qualified["bariatric_date"], errors="coerce")))

start = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"Scanning diagnosis.csv for K31.84, restricted to {len(qualified_ids):,} patients...")

last_K3184_date = {}
rows_seen = 0
chunk_num = 0

for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(qualified_ids)]
    if not chunk.empty:
        k = chunk[(chunk["code_system"] == "ICD-10-CM") & (chunk["code"] == "K31.84")]
        if not k.empty:
            for pid, d in k.groupby("patient_id")["date"].max().items():
                if pid not in last_K3184_date or d > last_K3184_date[pid]:
                    last_K3184_date[pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - start) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"  done - scanned {rows_seen:,} rows\n")

qualified["last_K31_84_date"] = qualified["patient_id"].map(last_K3184_date)
qualified["last_K31_84_date"] = pd.to_datetime(qualified["last_K31_84_date"], format="%Y%m%d", errors="coerce")

surgery_dt = pd.to_datetime(qualified["bariatric_date"], errors="coerce")
days_last_to_surgery = (surgery_dt - qualified["last_K31_84_date"]).dt.days
qualified["days_last_K3184_to_surgery"] = days_last_to_surgery

print("RESULTS:")
print(f"Mean days between most-recent K31.84 code and surgery:   {days_last_to_surgery.mean():,.0f} days (~{days_last_to_surgery.mean()/365.25:.1f} years)")
print(f"Median days between most-recent K31.84 code and surgery: {days_last_to_surgery.median():,.0f} days (~{days_last_to_surgery.median()/365.25:.1f} years)")

print("\nHow many had a K31.84 code within X time of surgery (evidence it was still being coded near surgery):")
for label, max_days in [("90 days", 90), ("6 months", 182), ("1 year", 365), ("2 years", 730)]:
    n = (days_last_to_surgery <= max_days).sum()
    print(f"  within {label}: {n:,} ({100*n/len(qualified):.1f}%)")

n_only_old = (days_last_to_surgery > 730).sum()
print(f"\n  more than 2 years before surgery (last K31.84 code is old): {n_only_old:,} ({100*n_only_old/len(qualified):.1f}%)")

qualified.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-start)/60:.1f} minutes")
