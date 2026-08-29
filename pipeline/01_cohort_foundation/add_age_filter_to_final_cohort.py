"""
add_age_filter_to_final_cohort.py

Extends cohort_closest_K3184_and_diabetes_to_surgery.csv (the 1,118-patient
cohort) with age-at-surgery, and applies the hard 18+ requirement.

IMPORTANT LIMITATION (same as before): patient.csv only has year_of_birth,
not a full date of birth - age is approximate, +/- up to 1 year depending
on whether the patient's actual birthday falls before or after surgery
within that calendar year.

Unlike the earlier age script (which only flagged, didn't exclude), this
APPLIES the exclusion - per the finalized rule, age >=18 at surgery is a
hard requirement, not just a flag for review.

Missing year_of_birth is treated as its own category (age_unknown), kept
separate from confirmed-under-18 - not silently auto-included or excluded,
since that's a real judgment call rather than something to decide silently.
"""

import subprocess
import pandas as pd

PATIENT_FILE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia/patient.csv"
INPUT_CSV = "cohort_closest_K3184_and_diabetes_to_surgery.csv"
OUTPUT_CSV = "final_cohort_with_age.csv"

print(">>> SCRIPT VERSION: add_age_filter_to_final_cohort_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"])
print(f"Cohort size: {len(cohort_ids):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
surgery_year = surgery_dt.dt.year

print(f"\nScanning patient.csv for year_of_birth, restricted to {len(cohort_ids):,} patients...")

proc = subprocess.Popen(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
yob_lookup = {}
rows_seen = 0
chunk_num = 0
for chunk in pd.read_csv(proc.stdout, usecols=["patient_id", "year_of_birth"], dtype=str, chunksize=500_000):
    chunk_num += 1
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        for pid, yob in zip(chunk["patient_id"], chunk["year_of_birth"]):
            if pd.notna(yob) and pid not in yob_lookup:  # first-row-wins for duplicates
                try:
                    yob_lookup[pid] = int(float(yob))
                except (ValueError, TypeError):
                    pass
    if chunk_num % 50 == 0:
        print(f"    ...{rows_seen:,} rows scanned")
proc.stdout.close()
proc.wait()

print(f"  done - scanned {rows_seen:,} rows")
print(f"  matched year_of_birth for {len(yob_lookup):,}/{len(cohort_ids):,} cohort patients")

df["year_of_birth"] = df["patient_id"].map(yob_lookup)
df["age_at_surgery_approx"] = surgery_year - df["year_of_birth"]

df["surgery_date_missing"] = surgery_dt.isna()
df["age_unknown"] = df["year_of_birth"].isna()
df["meets_age_requirement"] = df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] >= 18)

n_unknown = df["age_unknown"].sum()
n_surgery_missing = df["surgery_date_missing"].sum()
n_meets_age = df["meets_age_requirement"].sum()
n_under18_confirmed = (df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] < 18)).sum()

print(f"\nOf {len(df):,} cohort patients:")
print(f"  age unknown (no year_of_birth on record): {n_unknown:,}")
print(f"  surgery date missing (separate problem):   {n_surgery_missing:,}")
print(f"  confirmed under 18 at surgery (excluded):  {n_under18_confirmed:,}")
print(f"  meets age requirement (18+, included):      {n_meets_age:,}")

print("\nAge distribution (sanity check for outliers):")
print(df["age_at_surgery_approx"].describe())

if n_under18_confirmed > 0:
    print("\nExcluded for age (for reference):")
    cols = ["patient_id", "year_of_birth", "bariatric_date", "age_at_surgery_approx"]
    print(df[df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] < 18)][cols].to_string(index=False))

if n_unknown > 0:
    print(f"\n{n_unknown} patients have unknown age - NOT automatically excluded or included.")
    print("Review these separately before deciding how to handle them in the final count.")

# --- Final cohort: apply the age requirement ---
final_cohort = df[df["meets_age_requirement"]].copy()
print(f"\n{'='*70}")
print(f"FINAL COHORT (all 4 criteria, including age 18+): {len(final_cohort):,}")
print(f"{'='*70}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(df):,} rows - includes EVERYONE from the 1,118,")
print("with meets_age_requirement marking who passes the final filter.")
print("Nobody is physically removed from this file - filter on meets_age_requirement")
print("to get the final qualifying cohort, or use age_unknown to review those separately.)")
