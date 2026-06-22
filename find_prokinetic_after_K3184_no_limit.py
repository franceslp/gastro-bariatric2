"""
find_prokinetic_after_K3184_no_limit.py

For the master cohort file: finds the earliest prokinetic drug record on or
after closest_K31_84_strictly_before_surgery, with NO time window limit.

Same-day inclusive (prokinetic date >= K31.84 anchor date).
No upper bound going forward - captures any prokinetic ever recorded after
that diagnosis, not just within 1 year.

The 1-year window version is still kept separately for the exclusion sheet.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MED_FILE = f"{GCS_BASE}/medication_ingredient.csv"

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_prokinetic_after_K3184_no_limit.csv"

PROKINETIC_RXNORM_CODES = {
    "6915": "metoclopramide",
    "4053": "erythromycin",
    "3626": "domperidone",
    "2107310": "prucalopride",
}

print(">>> SCRIPT VERSION: find_prokinetic_after_K3184_no_limit_v1 <<<")

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


print(f"\nScanning medication_ingredient.csv for prokinetics on/after K31.84 anchor (no time limit)...")

earliest_by_drug = {name: {} for name in PROKINETIC_RXNORM_CODES.values()}
code_occurrence_counts = {code: 0 for code in PROKINETIC_RXNORM_CODES}

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(MED_FILE, usecols=["patient_id", "code_system", "code", "start_date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()
        rx = chunk[(chunk["code_system"] == "RxNorm") & (chunk["code"].isin(PROKINETIC_RXNORM_CODES.keys()))].copy()
        if not rx.empty:
            rx["start_date"] = pd.to_datetime(rx["start_date"], format="%Y%m%d", errors="coerce")
            rx = rx[rx["start_date"].notna()]
            if not rx.empty:
                for code, n in rx["code"].value_counts().items():
                    code_occurrence_counts[code] += n
                rx["anchor"] = rx["patient_id"].map(anchor_lookup)
                qualifying = rx[rx["anchor"].notna() & (rx["start_date"] >= rx["anchor"])].copy()
                if not qualifying.empty:
                    qualifying["drug"] = qualifying["code"].map(PROKINETIC_RXNORM_CODES)
                    for (pid, drug), d in qualifying.groupby(["patient_id", "drug"])["start_date"].min().items():
                        if pid not in earliest_by_drug[drug] or d < earliest_by_drug[drug][pid]:
                            earliest_by_drug[drug][pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")
print("\nRxNorm code occurrence counts (valid dates, cohort patients only):")
for code, name in PROKINETIC_RXNORM_CODES.items():
    print(f"  {code} ({name}): {code_occurrence_counts[code]:,}")

first_prokinetic_date = {}
first_prokinetic_drug = {}
for drug, pid_dates in earliest_by_drug.items():
    for pid, d in pid_dates.items():
        if pid not in first_prokinetic_date or d < first_prokinetic_date[pid]:
            first_prokinetic_date[pid] = d
            first_prokinetic_drug[pid] = drug

print(f"Missing K31.84 anchors: {df['closest_K31_84_strictly_before_surgery'].isna().sum():,}")

df["first_prokinetic_after_K3184_dx_drug"] = df["patient_id"].map(first_prokinetic_drug)
df["first_prokinetic_after_K3184_dx_date"] = df["patient_id"].map(first_prokinetic_date)

df["days_to_prokinetic_after_K3184"] = (
    pd.to_datetime(df["first_prokinetic_after_K3184_dx_date"], errors="coerce")
    - pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
).dt.days

n_qualifying = df["first_prokinetic_after_K3184_dx_date"].notna().sum()
print(f"\nPatients with any prokinetic on/after K31.84 anchor: {n_qualifying:,}/{len(df):,}")

print("\nWhich drug was earliest:")
print(df["first_prokinetic_after_K3184_dx_drug"].value_counts())

p_dt = pd.to_datetime(df["first_prokinetic_after_K3184_dx_date"], errors="coerce")
anchor_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")

same_day = (p_dt.notna() & anchor_dt.notna() & ((p_dt - anchor_dt).dt.days == 0)).sum()
print(f"Same-day prokinetic as K31.84 anchor: {same_day:,}")

print("\nDays to prokinetic after K31.84 diagnosis - summary:")
print(df["days_to_prokinetic_after_K3184"].describe())
for window in [30, 90, 180, 365]:
    n = (df["days_to_prokinetic_after_K3184"].notna() & (df["days_to_prokinetic_after_K3184"] <= window)).sum()
    print(f"  within {window} days: {n:,}/{n_qualifying:,}" if n_qualifying > 0 else f"  within {window} days: 0")

n_before = (p_dt.notna() & anchor_dt.notna() & (p_dt < anchor_dt)).sum()
print(f"\nProkinetic dated BEFORE anchor (should be 0): {n_before:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
