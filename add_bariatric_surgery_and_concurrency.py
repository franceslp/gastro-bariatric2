"""
add_bariatric_surgery_and_concurrency.py

Adds bariatric surgery (sleeve gastrectomy / Roux-en-Y gastric bypass)
evidence to the cohort, then flags patients meeting a diabetes-gastroparesis
concurrency rule relative to the surgery date: the later of
(first_diabetes_dx_date, first_K31_84_date) - the point at which both
conditions are first jointly documented - must fall within some window
before the bariatric surgery date.

WINDOW LENGTH: rather than locking in one cutoff, this generates BOTH a
1-year window (a deliberate, stricter alternative) and Sadda et al.'s own
published 5-year window as separate columns (meets_1yr_concurrency_rule,
meets_5yr_concurrency_rule) - costs nothing extra since
days_concurrent_to_bariatric_surgery is already continuous, and lets you
compare patient counts under each before committing to one with Dr. Sujka.
Add more entries to CONCURRENCY_WINDOWS_YEARS below for other cutoffs.

CPT codes (reused from the sister bariatric-surgery project, where they
were specifically chosen to capture sleeve/bypass and exclude other
bariatric procedures like banding or duodenal switch):
    43775   Laparoscopic sleeve gastrectomy
    43644   Laparoscopic gastric bypass (Roux-en-Y, short limb)
    43645   Laparoscopic gastric bypass with small bowel reconstruction
    43846   Open gastric bypass (short limb)
    43847   Open gastric bypass with small bowel reconstruction

ASSUMPTIONS (flagged explicitly, not silently decided):
  - No date-range restriction is applied to the surgery date itself. Sadda's
    own study window was 2010-2023; the sister bariatric project used
    2016+. Neither is applied here by default - add a min/max cutoff on
    bariatric_date below if you want to match either.
  - No age >= 18 filter is applied, since age/year_of_birth isn't merged
    into this cohort file - that would need a separate patient.csv pull.
  - "First qualifying bariatric surgery" = earliest date among ANY of the
    5 CPT codes. Sadda's concurrency rule itself structurally requires the
    diabetes+gastroparesis joint-documentation point to fall BEFORE the
    surgery date, so it naturally excludes patients whose bariatric
    surgery preceded their gastroparesis diagnosis - no separate ordering
    flag is needed for that.
  - Same-day sleeve+bypass codes are flagged (possible coding error /
    revisional surgery) but NOT excluded automatically.

Restricts the procedure.csv scan to cohort patients only, same pattern as
the diabetes script - filter to cohort_ids first, before the CPT check.

Run this AFTER add_erythromycin_routes_to_cohort.py. Writes a NEW file.
"""

import os
import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"
COHORT_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"

BARIATRIC_CPT_CODES = {"43775", "43644", "43645", "43846", "43847"}
SLEEVE_CODES = {"43775"}
BYPASS_CODES = {"43644", "43645", "43846", "43847"}
CONCURRENCY_WINDOWS_YEARS = {"1yr": 1, "5yr": 5}
CONCURRENCY_WINDOWS_DAYS = {
    label: round(years * 365.25) for label, years in CONCURRENCY_WINDOWS_YEARS.items()
}  # {"1yr": 365, "5yr": 1826}

if not os.path.exists(COHORT_CSV):
    raise FileNotFoundError(f"Missing required input file: {COHORT_CSV}")

print(f"Loading {COHORT_CSV} to get the cohort patient list...")
cohort_df = pd.read_csv(COHORT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = ["first_K31_84_date", "has_diabetes_dx", "first_diabetes_dx_date", "in_study_period"]
missing_cols = [c for c in required_cols if c not in cohort_df.columns]
if missing_cols:
    raise ValueError(f"{COHORT_CSV} is missing expected columns: {missing_cols}")

cohort_ids = set(cohort_df["patient_id"])
print(f"  cohort size: {len(cohort_ids):,} patients")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    cmd = ["gsutil", "cat", gcs_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning procedure.csv for bariatric surgery codes ({', '.join(sorted(BARIATRIC_CPT_CODES))}),")
print("restricted to the cohort's patient_ids...")

bariatric_first_date = {}
bariatric_codes_seen = {}
bariatric_dates_seen = {}

rows_seen = 0
chunk_num = 0
proc_start = time.time()
for chunk in stream_gcs_csv(PROCEDURE_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        mask = (chunk["code_system"] == "CPT") & (chunk["code"].isin(BARIATRIC_CPT_CODES))
        if mask.any():
            hits = chunk.loc[mask, ["patient_id", "code", "date"]].copy()
            hits["date"] = pd.to_datetime(hits["date"], format="%Y%m%d", errors="coerce")
            for pid, sub in hits.groupby("patient_id"):
                bariatric_codes_seen.setdefault(pid, set()).update(sub["code"])
                bariatric_dates_seen.setdefault(pid, set()).update(sub["date"].dropna())
                min_date = sub["date"].min()
                if pd.notna(min_date) and (
                    pid not in bariatric_first_date
                    or pd.isna(bariatric_first_date[pid])
                    or min_date < bariatric_first_date[pid]
                ):
                    bariatric_first_date[pid] = min_date

    if chunk_num % 20 == 0:
        elapsed_min = (time.time() - proc_start) / 60
        print(
            f"    ...procedure.csv: {rows_seen:,} rows processed so far "
            f"({elapsed_min:.1f} min elapsed, {len(bariatric_first_date):,} bariatric patients found)"
        )

print(f"\n  scanned {rows_seen:,} procedure rows")
print(f"  cohort patients with a qualifying bariatric surgery code (any year): {len(bariatric_first_date):,}")

cohort_df["has_bariatric_surgery"] = cohort_df["patient_id"].isin(bariatric_first_date)
cohort_df["bariatric_date"] = cohort_df["patient_id"].map(bariatric_first_date)
bariatric_code_types = {pid: ",".join(sorted(codes)) for pid, codes in bariatric_codes_seen.items()}
cohort_df["bariatric_cpt_codes_seen"] = cohort_df["patient_id"].map(bariatric_code_types)
cohort_df["num_distinct_bariatric_cpt_codes"] = cohort_df["patient_id"].map(
    {pid: len(codes) for pid, codes in bariatric_codes_seen.items()}
).fillna(0).astype(int)
# Distinct from the above - a revisional surgery could reuse the same CPT
# code on a different date, which num_distinct_bariatric_cpt_codes alone
# wouldn't catch (same lesson learned with GES distinct-codes vs
# distinct-dates earlier in this build).
cohort_df["num_distinct_bariatric_surgery_dates"] = cohort_df["patient_id"].map(
    {pid: len(dates) for pid, dates in bariatric_dates_seen.items()}
).fillna(0).astype(int)

# Flag (don't exclude) patients with both a sleeve and a bypass code on file.
cohort_df["has_both_sleeve_and_bypass_codes"] = cohort_df["bariatric_cpt_codes_seen"].apply(
    lambda s: isinstance(s, str)
    and bool(SLEEVE_CODES & set(s.split(",")))
    and bool(BYPASS_CODES & set(s.split(",")))
)

# --- Diabetes-gastroparesis concurrency rule, relative to surgery date --
# Adapted from Sadda et al.'s concept ("the first concurrent documentation
# of diabetes and gastroparesis", i.e. the later of the two first-diagnosis
# dates). Rather than locking in one window length, both 1-year (a
# deliberate stricter alternative discussed earlier) and 5-year (Sadda's
# own published window) flags are generated side by side, since
# days_concurrent_to_bariatric_surgery is already a continuous value -
# this costs nothing extra and lets you compare patient counts under each
# before committing to one with Dr. Sujka.
first_k3184_dt = pd.to_datetime(cohort_df["first_K31_84_date"], errors="coerce")
first_diabetes_dt = pd.to_datetime(cohort_df["first_diabetes_dx_date"], errors="coerce")
bariatric_dt = pd.to_datetime(cohort_df["bariatric_date"], errors="coerce")

both_dx_present = first_k3184_dt.notna() & first_diabetes_dt.notna()
# .max(axis=1) with default skipna=True would return whichever date exists
# even if the OTHER one is NaT - not what's wanted, since "concurrent"
# requires BOTH to be present. .where() forces NaT for rows that don't
# satisfy that, while keeping a clean datetime64 dtype throughout (avoiding
# the pd.NA/object-dtype pitfall caught earlier in the GES script).
raw_concurrent_dt = pd.concat([first_k3184_dt, first_diabetes_dt], axis=1).max(axis=1)
concurrent_dt = raw_concurrent_dt.where(both_dx_present, pd.NaT)
cohort_df["diabetes_gastroparesis_concurrent_date"] = concurrent_dt

all_required_present = both_dx_present & cohort_df["has_bariatric_surgery"] & bariatric_dt.notna()
cohort_df["days_concurrent_to_bariatric_surgery"] = (bariatric_dt - concurrent_dt).dt.days

for label, window_days in CONCURRENCY_WINDOWS_DAYS.items():
    cohort_df[f"meets_{label}_concurrency_rule"] = (
        all_required_present
        & (cohort_df["days_concurrent_to_bariatric_surgery"] >= 0)
        & (cohort_df["days_concurrent_to_bariatric_surgery"] <= window_days)
    )

print("\nSANITY CHECK:")
print(f"  cohort size: {len(cohort_df):,}")
print(f"  cohort patients with bariatric surgery: {cohort_df['has_bariatric_surgery'].sum():,}")
print(
    f"  of those, with both sleeve AND bypass codes on file (flagged, not excluded): "
    f"{cohort_df['has_both_sleeve_and_bypass_codes'].sum():,}"
)
print(
    f"  of those, with more than one distinct surgery date on record (possible revision): "
    f"{(cohort_df['has_bariatric_surgery'] & (cohort_df['num_distinct_bariatric_surgery_dates'] > 1)).sum():,}"
)
for label in CONCURRENCY_WINDOWS_DAYS:
    print(f"  cohort patients meeting the {label} concurrency rule: {cohort_df[f'meets_{label}_concurrency_rule'].sum():,}")

in_period = cohort_df["in_study_period"]
print(f"\nOf the {in_period.sum():,} in-study-period K31.84 patients:")
print(f"  have bariatric surgery on record: {(in_period & cohort_df['has_bariatric_surgery']).sum():,}")
for label in CONCURRENCY_WINDOWS_DAYS:
    n = (in_period & cohort_df[f"meets_{label}_concurrency_rule"]).sum()
    print(f"  meet the {label} concurrency rule: {n:,}")

cohort_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with has_bariatric_surgery, bariatric_date, bariatric_cpt_codes_seen,")
print("num_distinct_bariatric_cpt_codes, num_distinct_bariatric_surgery_dates,")
print("has_both_sleeve_and_bypass_codes, diabetes_gastroparesis_concurrent_date,")
print("days_concurrent_to_bariatric_surgery, meets_1yr_concurrency_rule, and")
print("meets_5yr_concurrency_rule columns added.")
print(f"(Original {COHORT_CSV} left untouched - use the new file as input to the next step.)")

total_elapsed_min = (time.time() - SCRIPT_START_TIME) / 60
print(f"\nTotal script runtime: {total_elapsed_min:.1f} minutes")
