"""
find_GES_before_closest_dx_1yr.py

For the cohort, finds the closest qualifying GES record:
  - on or BEFORE closest_K31_84_strictly_before_surgery (same-day included)
  - within 1 year (365 days) of that same anchor date

Single anchor only (closest-to-surgery K31.84), per confirmed policy -
this is the diagnosis most relevant to the surgical comparator-group
decision Dr. Sujka described. Same anchor, same 1-year window, same
same-day-inclusive rule as the companion prokinetic script - the only
difference is direction (before vs after).
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"  # has closest_K31_84_strictly_before_surgery
OUTPUT_CSV = "cohort_GES_before_closest_dx_1yr.csv"

GES_CPT_CODES = {"78264", "78265", "78266"}
WINDOW_DAYS = 365

print(">>> SCRIPT VERSION: find_GES_before_closest_dx_1yr_v1 <<<")

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


print(f"\nScanning procedure.csv for GES, restricted to {len(cohort_ids):,} patients...")

closest_GES_date = {}
closest_GES_code = {}
code_occurrence_counts = {code: 0 for code in GES_CPT_CODES}  # QA: confirm these codes actually occur

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(PROCEDURE_FILE, usecols=["patient_id", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        ges = chunk[chunk["code"].isin(GES_CPT_CODES)].copy()
        if not ges.empty:
            ges["date"] = pd.to_datetime(ges["date"], format="%Y%m%d", errors="coerce")
            ges = ges[ges["date"].notna()]
            if not ges.empty:
                for code, n in ges["code"].value_counts().items():
                    code_occurrence_counts[code] += n
                ges["anchor"] = ges["patient_id"].map(anchor_lookup)
                ges["days_before_anchor"] = (ges["anchor"] - ges["date"]).dt.days
                # same-day included (>=0), within 1 year (<=365)
                qualifying = ges[ges["anchor"].notna() & (ges["days_before_anchor"] >= 0) & (ges["days_before_anchor"] <= WINDOW_DAYS)]
                if not qualifying.empty:
                    # closest = latest date among qualifying (smallest days_before_anchor)
                    for pid, sub in qualifying.groupby("patient_id"):
                        d = sub["date"].max()
                        codes = sorted(sub.loc[sub["date"] == d, "code"].unique())
                        if pid not in closest_GES_date or d > closest_GES_date[pid]:
                            closest_GES_date[pid] = d
                            closest_GES_code[pid] = ",".join(codes)

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")
print("\nGES CPT code occurrence counts in this cohort (QA - confirms these codes actually occur):")
for code in GES_CPT_CODES:
    print(f"  {code}: {code_occurrence_counts[code]:,} raw rows seen")

df["GES_before_closest_dx_1yr_date"] = df["patient_id"].map(closest_GES_date)
df["GES_before_closest_dx_1yr_code"] = df["patient_id"].map(closest_GES_code)

# Days from GES to diagnosis - useful for the eventual write-up, mirrors
# days_to_prokinetic_after_K3184 in the companion script
df["days_GES_before_K3184"] = (
    pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
    - pd.to_datetime(df["GES_before_closest_dx_1yr_date"], errors="coerce")
).dt.days

n_qualifying = df["GES_before_closest_dx_1yr_date"].notna().sum()
print(f"\nPatients with a qualifying GES (on/before anchor, within 1yr): {n_qualifying:,}/{len(df):,}")

# Confirms the same-day-inclusive rule is actually exercised, not just
# theoretically allowed.
g_dt_pre = pd.to_datetime(df["GES_before_closest_dx_1yr_date"], errors="coerce")
anchor_dt_pre = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
same_day = (g_dt_pre.notna() & anchor_dt_pre.notna() & ((anchor_dt_pre - g_dt_pre).dt.days == 0)).sum()
print(f"Same-day GES (GES and diagnosis on the exact same date): {same_day:,}")

print("\nDays GES before diagnosis - summary:")
print(df["days_GES_before_K3184"].describe())
for window in [30, 90, 180, 365]:
    n_within = (df["days_GES_before_K3184"].notna() & (df["days_GES_before_K3184"] <= window)).sum()
    print(f"  within {window} days: {n_within:,}/{n_qualifying:,} ({100*n_within/n_qualifying:.1f}% of qualifying patients)" if n_qualifying > 0 else f"  within {window} days: 0")

# Defensive checks
g_dt = pd.to_datetime(df["GES_before_closest_dx_1yr_date"], errors="coerce")
anchor_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
n_after_anchor = (g_dt.notna() & anchor_dt.notna() & (g_dt > anchor_dt)).sum()
n_over_window = (g_dt.notna() & anchor_dt.notna() & ((anchor_dt - g_dt).dt.days > WINDOW_DAYS)).sum()
# Checks the DERIVED column's arithmetic directly, not just the raw date
# comparison above - catches a bug in the date-subtraction itself, even if
# the raw g_dt/anchor_dt comparison happened to pass.
n_negative_derived = (df["days_GES_before_K3184"] < 0).sum()
print(f"\nGES dated AFTER anchor (should be 0): {n_after_anchor:,}")
print(f"GES beyond the 1yr window (should be 0): {n_over_window:,}")
print(f"Negative days in the derived column (should be 0): {n_negative_derived:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
