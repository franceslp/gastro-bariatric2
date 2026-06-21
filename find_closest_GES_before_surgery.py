"""
find_closest_GES_before_surgery.py

For all 1,118 cohort patients (age filter NOT applied to this scan - same
reasoning as the E10/E11 script, age is a column to filter on later), finds
the closest GES (gastric emptying study) occurrence STRICTLY before surgery
(no same-day, matching K31.84/E10/E11 for consistency).

Captures both:
  - the date of that closest occurrence
  - the specific CPT code(s) used at that date (78264 / 78265 / 78266) -
    ties on the same date capture ALL codes seen that day, not an
    arbitrary pick (same fix already applied to the E10/E11 script)

No Oct-2015 floor applied - that constraint only ever applied to K31.84
(an ICD-10-CM diagnosis code tied to the ICD-9-to-10 transition). GES codes
are CPT procedure codes, unrelated to that transition entirely.

GES_code_types (already in the file) tracks codes used ANY TIME, no timing
restriction - this is a different, timing-specific question.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"

INPUT_CSV = "cohort_with_E10_E11_specific_codes.csv"
OUTPUT_CSV = "cohort_with_GES_before_surgery.csv"

GES_CPT_CODES = {"78264", "78265", "78266"}

print(">>> SCRIPT VERSION: find_closest_GES_before_surgery_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(df["patient_id"])
print(f"Full cohort size (all patients, age filter NOT applied to this scan): {len(cohort_ids):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
surgery_lookup = dict(zip(df["patient_id"], surgery_dt))

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning procedure.csv for GES codes strictly before surgery, restricted to {len(cohort_ids):,} patients...")

closest_GES = {}  # pid -> (date, comma-joined codes) of the closest qualifying occurrence

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
            ges["surgery_date"] = ges["patient_id"].map(surgery_lookup)
            strictly_before = ges["date"].notna() & ges["surgery_date"].notna() & (ges["date"] < ges["surgery_date"])
            qualifying = ges[strictly_before]
            if not qualifying.empty:
                for pid, sub in qualifying.groupby("patient_id"):
                    d = sub["date"].max()
                    codes = sorted(sub.loc[sub["date"] == d, "code"].unique())
                    if pid not in closest_GES or d > closest_GES[pid][0]:
                        closest_GES[pid] = (d, ",".join(codes))

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, {len(closest_GES):,} found so far)")

print(f"\n  done - scanned {rows_seen:,} rows")

df["closest_GES_date_before_surgery"] = df["patient_id"].map(
    lambda pid: closest_GES[pid][0] if pid in closest_GES else None
)
df["closest_GES_code_before_surgery"] = df["patient_id"].map(
    lambda pid: closest_GES[pid][1] if pid in closest_GES else None
)

n_with_ges = df["closest_GES_date_before_surgery"].notna().sum()
print(f"\nPatients with a GES strictly before surgery: {n_with_ges:,}/{len(df):,}")

# Defensive check: nobody should have a closest GES date on or after surgery,
# since the scan filtered to strictly-before only.
g_dt = pd.to_datetime(df["closest_GES_date_before_surgery"], errors="coerce")
n_not_strictly_before = (g_dt.notna() & surgery_dt.notna() & (g_dt >= surgery_dt)).sum()
print(f"GES dates on or after surgery (should be 0): {n_not_strictly_before:,}")

# Cross-check against first_GES_date (computed independently, much earlier in
# the project): closest_GES_date_before_surgery is, by definition, drawn from
# the same set of dates first_GES_date was minimized over - so it can never
# be earlier than that true minimum. A failure here would mean the two scans
# used inconsistent code-matching logic somewhere, the same category of bug
# caught earlier today with the erythromycin date-parsing mismatch.
if "first_GES_date" in df.columns:
    first_ges_dt = pd.to_datetime(df["first_GES_date"], errors="coerce")
    inconsistent = g_dt.notna() & first_ges_dt.notna() & (g_dt < first_ges_dt)
    n_inconsistent = inconsistent.sum()
    print(f"Patients where closest GES is somehow EARLIER than first_GES_date (should be 0): {n_inconsistent:,}")
    if n_inconsistent > 0:
        print("  ^ Investigate - the two GES scans may be using inconsistent code-matching logic.")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
