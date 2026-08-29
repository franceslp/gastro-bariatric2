"""
find_E10_E11_specific_codes.py

For the 1,117 final-cohort patients, finds:
  - ALL distinct E10/E11 specific codes ever seen (e.g. "E11.9, E11.65"),
    no timing restriction - a general "what diabetes codes does this
    patient have" list
  - the SPECIFIC code (not just the date) at the closest E10/E11
    occurrence strictly before surgery - same timing rule as before
    (strict before, no same-day)

Extends the diagnosis.csv work already done in Script 2 (which only
tracked dates, not the specific code values) - same scan target, new
piece of information captured.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

INPUT_CSV = "final_cohort_with_age.csv"
OUTPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"

print(">>> SCRIPT VERSION: find_E10_E11_specific_codes_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
# Scanning ALL 1,118 patients (not just the 1,117 who pass meets_age_requirement)
# - age is a column to filter on later, not a reason to withhold descriptive
# data from the one patient it excludes. Same pattern as Rao/Definition 1/
# the 5yr rule earlier today: compute everything, decide filtering later.
cohort_ids = set(df["patient_id"])
print(f"Full cohort size (all 1,118, age filter NOT applied to this scan): {len(cohort_ids):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
surgery_lookup = dict(zip(df["patient_id"], surgery_dt))

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning diagnosis.csv for E10/E11 specific codes, restricted to {len(cohort_ids):,} patients...")

all_codes_seen = {}  # pid -> set of all distinct E10/E11 codes ever seen
closest_before_surgery = {}  # pid -> (date, code) of the closest qualifying occurrence

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
        if not icd.empty:
            e10e11 = icd[icd["code"].str.startswith(("E10", "E11"), na=False)].copy()
            if not e10e11.empty:
                # All codes ever seen, no timing restriction
                for pid, code in zip(e10e11["patient_id"], e10e11["code"]):
                    all_codes_seen.setdefault(pid, set()).add(code)

                # Closest occurrence strictly before surgery
                e10e11["date"] = pd.to_datetime(e10e11["date"], format="%Y%m%d", errors="coerce")
                e10e11["surgery_date"] = e10e11["patient_id"].map(surgery_lookup)
                strictly_before = e10e11["date"].notna() & e10e11["surgery_date"].notna() & (e10e11["date"] < e10e11["surgery_date"])
                qualifying = e10e11[strictly_before]
                if not qualifying.empty:
                    for pid, sub in qualifying.groupby("patient_id"):
                        d = sub["date"].max()
                        # Capture ALL codes sharing this exact closest date, not
                        # just an arbitrary one - a patient coded with both
                        # E11.9 and E11.65 on the same visit is real, meaningful
                        # information, not something to silently collapse.
                        codes = sorted(sub.loc[sub["date"] == d, "code"].unique())
                        if pid not in closest_before_surgery or d > closest_before_surgery[pid][0]:
                            closest_before_surgery[pid] = (d, ",".join(codes))

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")

df["E10_E11_codes_seen"] = df["patient_id"].map(
    lambda pid: ", ".join(sorted(all_codes_seen[pid])) if pid in all_codes_seen else None
)
df["closest_E10_E11_code_before_surgery"] = df["patient_id"].map(
    lambda pid: closest_before_surgery[pid][1] if pid in closest_before_surgery else None
)
df["closest_E10_E11_date_before_surgery_v2"] = df["patient_id"].map(
    lambda pid: closest_before_surgery[pid][0] if pid in closest_before_surgery else None
)

n_with_codes = df["E10_E11_codes_seen"].notna().sum()
n_with_closest = df["closest_E10_E11_code_before_surgery"].notna().sum()
print(f"\nPatients with E10/E11 codes recorded (any time): {n_with_codes:,}/{len(df):,}")
print(f"Patients with a closest E10/E11 code+date before surgery: {n_with_closest:,}/{len(df):,}")

# Hard check: this should match the existing closest_E10_or_E11_strictly_before_surgery
# date from Script 2, since it's the same population, same logic, just adding code value.
if "closest_E10_or_E11_strictly_before_surgery" in df.columns:
    existing_dt = pd.to_datetime(df["closest_E10_or_E11_strictly_before_surgery"], errors="coerce")
    new_dt = pd.to_datetime(df["closest_E10_E11_date_before_surgery_v2"], errors="coerce")
    mismatch = (existing_dt.notna() | new_dt.notna()) & (existing_dt != new_dt)
    n_mismatch = mismatch.sum()
    print(f"\nMismatches vs. Script 2's existing closest-date column (should be 0): {n_mismatch:,}")
    if n_mismatch > 0:
        print("  ^ Investigate before trusting either column.")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
