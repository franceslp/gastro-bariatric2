"""
add_demographics.py

Pulls demographic columns from patient.csv for the 1,118-patient cohort:
  sex, race, ethnicity, marital_status, patient_regional_location,
  month_year_death (used to derive a deceased flag)

year_of_birth is already in final_cohort_with_age.csv so not re-pulled here.
patient.csv is only ~2.1 million rows so this runs fast.
"""

import subprocess
import pandas as pd

PATIENT_FILE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia/patient.csv"
INPUT_CSV = "final_cohort_with_age.csv"
OUTPUT_CSV = "cohort_with_demographics.csv"

DEMO_COLS = ["patient_id", "sex", "race", "ethnicity", "marital_status", "month_year_death"]

print(">>> SCRIPT VERSION: add_demographics_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"].dropna())
print(f"Cohort size: {len(cohort_ids):,} patients")

print(f"\nScanning patient.csv for demographics, restricted to {len(cohort_ids):,} patients...")

proc = subprocess.Popen(["gsutil", "cat", PATIENT_FILE], stdout=subprocess.PIPE)
demo_records = {}
duplicate_patient_rows = 0
rows_seen = 0
for chunk in pd.read_csv(proc.stdout, usecols=DEMO_COLS, dtype=str, chunksize=500_000):
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        for _, row in chunk.iterrows():
            pid = row["patient_id"]
            if pid in demo_records:
                duplicate_patient_rows += 1
            else:
                demo_records[pid] = row.to_dict()
proc.stdout.close()
proc.wait()
print(f"  done - scanned {rows_seen:,} rows")
print(f"  matched demographics for {len(demo_records):,}/{len(cohort_ids):,} cohort patients")
missing_demo = len(cohort_ids) - len(demo_records)
print(f"  missing demographic records: {missing_demo:,}")
print(f"  duplicate patient rows encountered in patient.csv: {duplicate_patient_rows:,}")

demo_df = pd.DataFrame.from_dict(demo_records, orient="index").reset_index(drop=True)
demo_df["patient_id"] = demo_df["patient_id"].astype(str)

# Derive a clean deceased flag from month_year_death
demo_df["deceased"] = demo_df["month_year_death"].notna() & (demo_df["month_year_death"].str.strip() != "")

df = df.merge(demo_df, on="patient_id", how="left")

n_unique = df["patient_id"].nunique()
if n_unique != len(df):
    raise ValueError(f"Merge produced {len(df):,} rows but only {n_unique:,} unique patient_ids.")
assert len(df) == 1118, f"Row count changed after merge: expected 1,118, got {len(df):,}"
print(f"Row count confirmed: {len(df):,} (no rows added or dropped by the merge)")

print(f"\nDemographic summary:")
print(f"\nSex:")
print(df["sex"].value_counts(dropna=False))
print(f"\nRace:")
print(df["race"].value_counts(dropna=False))
print(f"\nEthnicity:")
print(df["ethnicity"].value_counts(dropna=False))
print(f"\nMarital status:")
print(df["marital_status"].value_counts(dropna=False))
print(f"\nDeceased: {df['deceased'].sum():,}/{len(df):,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
