"""
find_prokinetic_after_closest_dx_1yr.py

For the cohort, finds the closest qualifying prokinetic drug record:
  - on or AFTER closest_K31_84_strictly_before_surgery (same-day included)
  - within 1 year (365 days) of that same anchor date

Single anchor only (closest-to-surgery K31.84), per confirmed policy -
this is the diagnosis most relevant to the surgical comparator-group
decision Dr. Sujka described.

Captures both the date AND which specific drug, same as prior scripts.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MED_INGREDIENT_FILE = f"{GCS_BASE}/medication_ingredient.csv"

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"  # has closest_K31_84_strictly_before_surgery
OUTPUT_CSV = "cohort_prokinetic_after_closest_dx_1yr.csv"

PROKINETIC_RXNORM_CODES = {
    "6915": "metoclopramide",
    "4053": "erythromycin",
    "3626": "domperidone",
    "2107310": "prucalopride",
}

WINDOW_DAYS = 365

print(">>> SCRIPT VERSION: find_prokinetic_after_closest_dx_1yr_v1 <<<")

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


print(f"\nScanning medication_ingredient.csv for prokinetics, restricted to {len(cohort_ids):,} patients...")

# Tracked per-drug so we know which specific drug was the earliest
# qualifying one. Named "earliest" not "closest" - since every qualifying
# date is constrained to be on/after the anchor, taking the minimum date
# IS the closest one (no direction ambiguity), but the name should say
# what the logic actually does.
earliest_by_drug = {name: {} for name in PROKINETIC_RXNORM_CODES.values()}
code_occurrence_counts = {code: 0 for code in PROKINETIC_RXNORM_CODES}  # QA: confirm these codes actually occur in this cohort

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(MED_INGREDIENT_FILE, usecols=["patient_id", "code_system", "code", "start_date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()
        rx = chunk[(chunk["code_system"] == "RxNorm") & (chunk["code"].isin(PROKINETIC_RXNORM_CODES.keys()))].copy()
        if not rx.empty:
            for code, n in rx["code"].value_counts().items():
                code_occurrence_counts[code] += n
            rx["start_date"] = pd.to_datetime(rx["start_date"], format="%Y%m%d", errors="coerce")
            rx = rx[rx["start_date"].notna()]
            if not rx.empty:
                rx["anchor"] = rx["patient_id"].map(anchor_lookup)
                rx["days_after_anchor"] = (rx["start_date"] - rx["anchor"]).dt.days
                # same-day included (>=0), within 1 year (<=365)
                qualifying = rx[rx["anchor"].notna() & (rx["days_after_anchor"] >= 0) & (rx["days_after_anchor"] <= WINDOW_DAYS)]
                if not qualifying.empty:
                    qualifying["drug"] = qualifying["code"].map(PROKINETIC_RXNORM_CODES)
                    for (pid, drug), d in qualifying.groupby(["patient_id", "drug"])["start_date"].min().items():
                        if pid not in earliest_by_drug[drug] or d < earliest_by_drug[drug][pid]:
                            earliest_by_drug[drug][pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")
print("\nRxNorm code occurrence counts in this cohort (QA - confirms these codes actually occur):")
for code, name in PROKINETIC_RXNORM_CODES.items():
    print(f"  {code} ({name}): {code_occurrence_counts[code]:,} raw rows seen")

# Collapse per-drug results down to one earliest-qualifying date+drug per patient
first_prokinetic_date = {}
first_prokinetic_drug = {}
for drug, pid_dates in earliest_by_drug.items():
    for pid, d in pid_dates.items():
        if pid not in first_prokinetic_date or d < first_prokinetic_date[pid]:
            first_prokinetic_date[pid] = d
            first_prokinetic_drug[pid] = drug

df["prokinetic_after_closest_dx_1yr_date"] = df["patient_id"].map(first_prokinetic_date)
df["prokinetic_after_closest_dx_1yr_drug"] = df["patient_id"].map(first_prokinetic_drug)

# Days from anchor to prokinetic - useful for the eventual write-up
# (median days to treatment, proportion within common windows)
df["days_to_prokinetic_after_K3184"] = (
    pd.to_datetime(df["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
    - pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
).dt.days

n_qualifying = df["prokinetic_after_closest_dx_1yr_date"].notna().sum()
print(f"\nPatients with a qualifying prokinetic (on/after anchor, within 1yr): {n_qualifying:,}/{len(df):,}")

print("\nWhich drug was the earliest qualifying one:")
print(df["prokinetic_after_closest_dx_1yr_drug"].value_counts())

print("\nDays to prokinetic after diagnosis - summary:")
print(df["days_to_prokinetic_after_K3184"].describe())
for window in [30, 90, 180, 365]:
    n_within = (df["days_to_prokinetic_after_K3184"] <= window).sum()
    print(f"  within {window} days: {n_within:,}/{n_qualifying:,} ({100*n_within/n_qualifying:.1f}% of qualifying patients)" if n_qualifying > 0 else f"  within {window} days: 0")

# Defensive checks
p_dt = pd.to_datetime(df["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
anchor_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
n_before_anchor = (p_dt.notna() & anchor_dt.notna() & (p_dt < anchor_dt)).sum()
n_over_window = (p_dt.notna() & anchor_dt.notna() & ((p_dt - anchor_dt).dt.days > WINDOW_DAYS)).sum()
print(f"\nProkinetic dated BEFORE anchor (should be 0): {n_before_anchor:,}")
print(f"Prokinetic beyond the 1yr window (should be 0): {n_over_window:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
