"""
find_GES_before_K3184_no_limit.py

For the master cohort file: finds the closest GES occurrence on or before
closest_K31_84_strictly_before_surgery, with NO time window limit.

Same-day inclusive (GES date <= K31.84 anchor date).
No upper bound going back - captures any GES ever recorded before that
diagnosis, not just within 1 year.

The 1-year window version is still kept separately for the exclusion sheet.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_GES_before_K3184_no_limit.csv"

GES_CPT_CODES = {"78264", "78265", "78266"}

print(">>> SCRIPT VERSION: find_GES_before_K3184_no_limit_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"].dropna())
print(f"Cohort size: {len(cohort_ids):,} patients")

anchor_lookup = dict(zip(df["patient_id"], pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")))
n_with_anchor = sum(1 for v in anchor_lookup.values() if pd.notna(v))
print(f"Patients with a valid anchor date: {n_with_anchor:,}/{len(df):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning procedure.csv for GES on/before K31.84 anchor (no time limit)...")

closest_GES_date = {}
closest_GES_code = {}
code_occurrence_counts = {code: 0 for code in GES_CPT_CODES}
whitespace_recovered = 0

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(PROCEDURE_FILE, usecols=["patient_id", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        raw_match = chunk["code"].isin(GES_CPT_CODES)
        chunk["code"] = chunk["code"].str.strip()
        stripped_match = chunk["code"].isin(GES_CPT_CODES)
        whitespace_recovered += (stripped_match & ~raw_match).sum()

        ges = chunk[stripped_match].copy()
        if not ges.empty:
            ges["date"] = pd.to_datetime(ges["date"], format="%Y%m%d", errors="coerce")
            ges = ges[ges["date"].notna()]
            if not ges.empty:
                for code, n in ges["code"].value_counts().items():
                    code_occurrence_counts[code] += n
                ges["anchor"] = ges["patient_id"].map(anchor_lookup)
                # same-day inclusive (<=), no lower bound on how far back
                qualifying = ges[ges["anchor"].notna() & (ges["date"] <= ges["anchor"])]
                if not qualifying.empty:
                    for pid, sub in qualifying.groupby("patient_id"):
                        d = sub["date"].max()
                        codes = sorted(sub.loc[sub["date"] == d, "code"].unique())
                        if pid not in closest_GES_date or d > closest_GES_date[pid]:
                            closest_GES_date[pid] = d
                            closest_GES_code[pid] = ",".join(codes)

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, {len(closest_GES_date):,} found so far)")

print(f"\n  done - scanned {rows_seen:,} rows")
print(f"  whitespace impact: {whitespace_recovered:,} rows only matched after stripping")
print("\nGES CPT code counts (valid dates only):")
for code in GES_CPT_CODES:
    print(f"  {code}: {code_occurrence_counts[code]:,}")

print(f"Missing K31.84 anchors: {df['closest_K31_84_strictly_before_surgery'].isna().sum():,}")

df["closest_GES_before_K3184_dx_code"] = df["patient_id"].map(closest_GES_code)
df["closest_GES_before_K3184_dx_date"] = df["patient_id"].map(closest_GES_date)

df["days_GES_before_K3184"] = (
    pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
    - pd.to_datetime(df["closest_GES_before_K3184_dx_date"], errors="coerce")
).dt.days

n_with_ges = df["closest_GES_before_K3184_dx_date"].notna().sum()
print(f"\nPatients with any GES on/before K31.84 anchor: {n_with_ges:,}/{len(df):,}")

g_dt = pd.to_datetime(df["closest_GES_before_K3184_dx_date"], errors="coerce")
anchor_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")

same_day = (g_dt.notna() & anchor_dt.notna() & ((anchor_dt - g_dt).dt.days == 0)).sum()
print(f"Same-day GES as K31.84 anchor: {same_day:,}")

print("\nDays GES before K31.84 diagnosis - summary:")
print(df["days_GES_before_K3184"].describe())
for window in [30, 90, 180, 365]:
    n = (df["days_GES_before_K3184"].notna() & (df["days_GES_before_K3184"] <= window)).sum()
    print(f"  within {window} days: {n:,}/{n_with_ges:,}" if n_with_ges > 0 else f"  within {window} days: 0")

n_after = (g_dt.notna() & anchor_dt.notna() & (g_dt > anchor_dt)).sum()
print(f"\nGES dated AFTER anchor (should be 0): {n_after:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
