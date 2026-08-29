"""
find_comparator_bariatric_patients.py

STAGE 1 of 3 for building the PSM comparator group.

Scans procedure.csv to find ALL patients who had a qualifying bariatric
surgery between Oct 1, 2015 and May 23, 2025, who are NOT in the
gastroparesis cohort (full 335,846 population excluded, not just the 110).

This is the raw comparator pool - gastroparesis exclusion and demographics
are applied in subsequent stages.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "comparator_bariatric_candidates.csv"

BARIATRIC_CPT_CODES = {"43775", "43644", "43645", "43846", "43847"}
STUDY_START = pd.Timestamp("2015-10-01")
STUDY_END = pd.Timestamp("2025-05-23")

print(">>> SCRIPT VERSION: find_comparator_bariatric_patients_v1 <<<")

# Load the full gastroparesis population to exclude
gp_df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False,
                     usecols=["patient_id"])
gp_ids = set(gp_df["patient_id"].dropna())
print(f"Gastroparesis population to exclude: {len(gp_ids):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning procedure.csv for bariatric surgery patients in study window...")
print(f"  Window: {STUDY_START.date()} to {STUDY_END.date()}")
print(f"  Excluding {len(gp_ids):,} gastroparesis patients")

# For each non-GP patient, find their earliest qualifying bariatric surgery
# in the study window
bariatric_dates = {}  # pid -> earliest surgery date in window
bariatric_codes = {}  # pid -> set of CPT codes seen

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(PROCEDURE_FILE, usecols=["patient_id", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    # Exclude gastroparesis patients immediately, and drop any rows with
    # missing patient_id (NaN IDs could survive into the output dict otherwise)
    chunk = chunk[chunk["patient_id"].notna()]
    chunk = chunk[~chunk["patient_id"].isin(gp_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()

        # One-time CPT format check on the first chunk - confirms codes aren't
        # being read as floats (e.g. "43775.0") which would silently miss them
        if chunk_num == 1:
            print(f"  CPT code sample from first chunk: {chunk['code'].value_counts().head(5).to_dict()}")
        hits = chunk[chunk["code"].isin(BARIATRIC_CPT_CODES)].copy()
        if not hits.empty:
            # No explicit format string - handles both YYYYMMDD and ISO dates safely
            hits["date"] = pd.to_datetime(hits["date"], errors="coerce")
            in_window = hits["date"].notna() & (hits["date"] >= STUDY_START) & (hits["date"] <= STUDY_END)
            hits = hits[in_window]
            if not hits.empty:
                for pid, grp in hits.groupby("patient_id"):
                    d = grp["date"].min()
                    # Only capture codes from the index surgery date, not all
                    # surgeries ever - avoids misleadingly mixing codes from
                    # separate surgical episodes into one "procedure type"
                    codes_at_index = set(grp.loc[grp["date"] == d, "code"].unique())
                    if pid not in bariatric_dates or d < bariatric_dates[pid]:
                        bariatric_dates[pid] = d
                        bariatric_codes[pid] = codes_at_index
                    elif d == bariatric_dates[pid]:
                        # same index date found in a different chunk - accumulate
                        bariatric_codes[pid].update(codes_at_index)

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, {len(bariatric_dates):,} candidates found)")

print(f"\n  done - scanned {rows_seen:,} rows")
print(f"  Non-gastroparesis bariatric surgery candidates found: {len(bariatric_dates):,}")

out_df = pd.DataFrame({
    "patient_id": list(bariatric_dates.keys()),
    "bariatric_date": [bariatric_dates[p] for p in bariatric_dates],
    "bariatric_cpt_codes_seen": [",".join(sorted(bariatric_codes[p])) for p in bariatric_dates],
})

out_df.to_csv(OUTPUT_CSV, index=False)

assert out_df["patient_id"].is_unique, "Duplicate patient_ids in output - investigate before PSM"
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows, all patient_ids unique confirmed)")

print("\nDate range sanity check (should be within study window):")
print(f"  Earliest surgery: {out_df['bariatric_date'].min()}")
print(f"  Latest surgery:   {out_df['bariatric_date'].max()}")

print("\nCPT code distribution (sanity check - sleeve should dominate):")
print(out_df["bariatric_cpt_codes_seen"].value_counts().head(20))

multi_cpt = out_df["bariatric_cpt_codes_seen"].str.contains(",", na=False).sum()
print(f"\nPatients with multiple CPT codes on index surgery date: {multi_cpt:,}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
print("\nNext: run find_comparator_exclude_gastroparesis.py to remove any")
print("patients with gastroparesis codes missed in the first pass.")
