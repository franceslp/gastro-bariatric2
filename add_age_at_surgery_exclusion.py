"""
add_age_at_surgery_exclusion.py

Pulls year_of_birth from patient.csv (never scanned before in this project)
for the 1,070-patient master cohort, and computes an APPROXIMATE age at
surgery: surgery_year - year_of_birth.

IMPORTANT LIMITATION: patient.csv only has year_of_birth, not a full date of
birth (standard for de-identified data like this). This means age is only
approximate, +/- up to 1 year, depending on whether the patient's actual
birthday falls before or after their surgery date within that calendar year.
A patient computed as "exactly 18" could in reality be anywhere from 17 to
18 years old at the actual moment of surgery. Worth keeping in mind for any
borderline cases - this isn't a precision instrument.

Patients with missing year_of_birth (reason_yob_missing populated) are
flagged separately as "age unknown" rather than silently excluded or
included - that's a judgment call for you, not something this script
should decide silently.
"""

import subprocess
import pandas as pd

PATIENT_FILE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia/patient.csv"
INPUT_CSV = "master_cohort_K3184_diabetes_before_surgery.csv"
OUTPUT_CSV = "master_cohort_K3184_diabetes_before_surgery_WITH_age.csv"

print("Loading master cohort...")
df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"])
print(f"  cohort size: {len(cohort_ids):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
surgery_year = surgery_dt.dt.year

print(f"\nScanning patient.csv for year_of_birth, restricted to {len(cohort_ids):,} patients...")

proc = subprocess.Popen(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
yob_lookup = {}
reason_missing_lookup = {}
rows_seen = 0
chunk_num = 0
for chunk in pd.read_csv(proc.stdout, usecols=["patient_id", "year_of_birth", "reason_yob_missing"],
                          dtype=str, chunksize=500_000):
    chunk_num += 1
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        if chunk["patient_id"].duplicated().any():
            print("  ^ NOTE: duplicate patient_ids found in this chunk of patient.csv (last row wins)")
        for pid, yob, reason in zip(chunk["patient_id"], chunk["year_of_birth"], chunk["reason_yob_missing"]):
            if pd.notna(yob) and pid not in yob_lookup:
                try:
                    yob_lookup[pid] = int(float(yob))
                except (ValueError, TypeError):
                    pass  # malformed year_of_birth value (e.g. "unknown") - leave unmatched, will show as age_unknown
            if pd.notna(reason):
                reason_missing_lookup[pid] = reason
    if chunk_num % 50 == 0:
        print(f"    ...{rows_seen:,} rows scanned")
proc.stdout.close()
proc.wait()

print(f"  done - scanned {rows_seen:,} rows")
print(f"  matched year_of_birth for {len(yob_lookup):,}/{len(cohort_ids):,} cohort patients")

df["year_of_birth"] = df["patient_id"].map(yob_lookup)
df["reason_yob_missing"] = df["patient_id"].map(reason_missing_lookup)
df["age_at_surgery_approx"] = surgery_year - df["year_of_birth"]

df["surgery_date_missing"] = surgery_dt.isna()
df["age_unknown"] = df["year_of_birth"].isna()
df["age_under_18_at_surgery"] = df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] < 18)
# Boundary flag: since only birth YEAR is available (not full DOB), a
# computed age of exactly 18 could really be 17 if the surgery happened
# before that year's birthday. <=18 catches these borderline cases for
# manual review, rather than confidently treating "18" as definitely adult.
df["possibly_under_18_at_surgery"] = df["age_at_surgery_approx"].notna() & (df["age_at_surgery_approx"] <= 18)

n_unknown = df["age_unknown"].sum()
n_surgery_missing = df["surgery_date_missing"].sum()
n_under18 = df["age_under_18_at_surgery"].sum()
n_possibly_under18 = df["possibly_under_18_at_surgery"].sum()
# Confirmed-adult count now correctly EXCLUDES boundary cases (age exactly
# 18) rather than silently lumping them in as "known 18+" - those patients
# are flagged separately above and need manual review, not an automatic
# adult classification.
n_known_18plus = (df["age_at_surgery_approx"].notna() & ~df["possibly_under_18_at_surgery"]).sum()

print(f"\nOf {len(df):,} cohort patients:")
print(f"  age unknown (no year_of_birth on record): {n_unknown:,}")
print(f"  surgery date missing (separate problem):   {n_surgery_missing:,}")
print(f"  under 18 at surgery (approx, strict <18):  {n_under18:,}")
print(f"  possibly under 18 (<=18, boundary cases for manual review): {n_possibly_under18:,}")
print(f"  confirmed adult (>18, no boundary ambiguity): {n_known_18plus:,}")

print("\nAge distribution (sanity check for outliers - e.g. min=12 or max=94 would be a red flag):")
print(df["age_at_surgery_approx"].describe())

if n_possibly_under18 > 0:
    print("\n  Possibly-under-18 patients (age <=18, for manual review):")
    print(df[df["possibly_under_18_at_surgery"]][["patient_id", "year_of_birth", "bariatric_date", "age_at_surgery_approx"]].to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with year_of_birth, age_at_surgery_approx, surgery_date_missing,")
print("age_unknown, age_under_18_at_surgery, and possibly_under_18_at_surgery columns.")
print("(Nobody was removed from the file - this flags candidates for exclusion, doesn't apply it.)")
print(" Decide separately whether to drop the under-18 and/or age-unknown rows.")
