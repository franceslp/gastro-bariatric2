"""
explore_K3184_full_history.py

Before deciding which K31.84 date should anchor GES/prokinetic checks,
this shows the FULL K31.84 encounter history for each of the 1,118 cohort
patients - every distinct date, not just first/last - plus summary stats
on how spread out these histories typically are.

This directly informs the anchor-date decision: if most patients have
their K31.84 dates tightly clustered, the first-vs-closest-vs-earliest-
valid distinction barely matters. If many patients have widely spread
histories (years between first and last occurrence), the choice matters
a lot.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "final_cohort_with_age.csv"
OUTPUT_CSV = "cohort_K3184_full_history.csv"

print(">>> SCRIPT VERSION: explore_K3184_full_history_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"])
print(f"Cohort size: {len(cohort_ids):,} patients")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for ALL K31.84 dates, restricted to {len(cohort_ids):,} patients...")

all_dates_per_patient = {}
whitespace_recovered_count = 0  # rows that only matched AFTER stripping - same measurement as the GES script
rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        raw_match = (chunk["code"] == "K31.84") & (chunk["code_system"] == "ICD-10-CM")
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()
        stripped_match = (chunk["code"] == "K31.84") & (chunk["code_system"] == "ICD-10-CM")
        newly_recovered = (stripped_match & ~raw_match).sum()
        if newly_recovered > 0:
            whitespace_recovered_count += newly_recovered

        k31 = chunk[stripped_match].copy()
        if not k31.empty:
            k31["date"] = pd.to_datetime(k31["date"], format="%Y%m%d", errors="coerce")
            k31 = k31[k31["date"].notna()]
            for pid, dates in k31.groupby("patient_id")["date"]:
                all_dates_per_patient.setdefault(pid, set()).update(dates.tolist())

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")
print(f"\nWHITESPACE IMPACT: {whitespace_recovered_count:,} rows only matched K31.84 AFTER stripping.")
if whitespace_recovered_count == 0:
    print("  -> ZERO recovered. Second independent confirmation (after the GES check) that")
    print("     whitespace is not an issue in this dataset's code columns.")
else:
    print(f"  -> NOT ZERO. Combined with any GES finding, this is a real, serious issue")
    print(f"     requiring a broader re-audit of every prior script's code-matching.")

# Build per-patient summary
records = []
for pid in sorted(cohort_ids):
    dates = sorted(all_dates_per_patient.get(pid, []))
    if dates:
        date_strs = ", ".join(d.strftime("%Y-%m-%d") for d in dates)
        span_days = (dates[-1] - dates[0]).days
    else:
        date_strs = None
        span_days = None
    records.append({
        "patient_id": pid,
        "num_distinct_K31_84_dates": len(dates),
        "all_K31_84_dates": date_strs,
        "first_to_last_span_days": span_days,
    })

history_df = pd.DataFrame(records)
df = df.merge(history_df, on="patient_id", how="left")

print("\n" + "="*70)
print("SUMMARY: how many distinct K31.84 dates does each patient have?")
print("="*70)
print(df["num_distinct_K31_84_dates"].value_counts().sort_index())

print("\n" + "="*70)
print("SUMMARY: span between first and last K31.84 date (days)")
print("="*70)
print(df["first_to_last_span_days"].describe())

print("\nHow many patients have a span > 365 days (more than a year between first and last)?")
n_spread_out = (df["first_to_last_span_days"] > 365).sum()
print(f"  {n_spread_out:,}/{len(df):,} ({100*n_spread_out/len(df):.1f}%)")

print("\nHow many patients have a span > 1825 days (more than 5 years)?")
n_very_spread = (df["first_to_last_span_days"] > 1825).sum()
print(f"  {n_very_spread:,}/{len(df):,} ({100*n_very_spread/len(df):.1f}%)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} - includes the full date list per patient for direct review.")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
