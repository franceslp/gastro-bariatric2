"""
find_closest_K3184_and_diabetes_to_surgery.py

For the corrected cohort, finds:
  - the K31.84 occurrence CLOSEST to (and STRICTLY BEFORE) the surgery
    date, on/after Oct 1, 2015 (matches the gating criterion exactly -
    a pre-Oct-2015 K31.84 occurrence is not a valid diagnosis for this
    project and should never be reported as "the closest one")
  - the E10 occurrence closest to (and strictly before) surgery
  - the E11 occurrence closest to (and strictly before) surgery
  - the closer of E10/E11 combined (whichever of the two is more recent)
  - a clean diabetes type label (Type 1 / Type 2 / Both / neither)

E10/E11 do NOT have the Oct 2015 floor applied - that constraint has only
ever been requested for K31.84 specifically, not diabetes codes.

Combined into ONE scan of diagnosis.csv (1.3 billion rows) rather than two
separate passes, since K31.84 and E10/E11 are both ICD-10-CM codes in the
same file.

STRICT BEFORE (date < surgery_date, same-day NOT counted) - confirmed:
diagnoses need to predate surgery, not coincide with it on the same day.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

STUDY_START = pd.Timestamp("2015-10-01")
INPUT_CSV = "funnel_v2_before_age.csv"
OUTPUT_CSV = "cohort_closest_K3184_and_diabetes_to_surgery.csv"

print(">>> SCRIPT VERSION: find_closest_K3184_and_diabetes_to_surgery_v2 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
df = df[df["passes_K3184_E10E11_gating"] == True].copy()
cohort_ids = set(df["patient_id"])
print(f"Cohort size (already passing K31.84 + E10/E11 gating): {len(cohort_ids):,} patients")

surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")
surgery_lookup = dict(zip(df["patient_id"], surgery_dt))

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print("\nScanning diagnosis.csv for K31.84, E10, and E11 - closest occurrence strictly before surgery -")
print(f"restricted to {len(cohort_ids):,} patients...")

closest_K3184 = {}
closest_E10 = {}
closest_E11 = {}
rows_seen = 0
chunk_num = 0

for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
        if not icd.empty:
            icd["date"] = pd.to_datetime(icd["date"], format="%Y%m%d", errors="coerce")
            icd["surgery_date"] = icd["patient_id"].map(surgery_lookup)
            strictly_before_surgery = icd["date"].notna() & icd["surgery_date"].notna() & (icd["date"] < icd["surgery_date"])
            icd = icd[strictly_before_surgery]

            if not icd.empty:
                k3184_hits = icd[icd["code"] == "K31.84"]
                # K31.84-specific: also requires on/after Oct 1 2015, matching
                # the gating criterion exactly. E10/E11 below do NOT get this
                # floor - it's only ever been requested for K31.84.
                k3184_hits = k3184_hits[k3184_hits["date"] >= STUDY_START]
                if not k3184_hits.empty:
                    for pid, d in k3184_hits.groupby("patient_id")["date"].max().items():
                        if pid not in closest_K3184 or d > closest_K3184[pid]:
                            closest_K3184[pid] = d

                e10_hits = icd[icd["code"].str.startswith("E10", na=False)]
                if not e10_hits.empty:
                    for pid, d in e10_hits.groupby("patient_id")["date"].max().items():
                        if pid not in closest_E10 or d > closest_E10[pid]:
                            closest_E10[pid] = d

                e11_hits = icd[icd["code"].str.startswith("E11", na=False)]
                if not e11_hits.empty:
                    for pid, d in e11_hits.groupby("patient_id")["date"].max().items():
                        if pid not in closest_E11 or d > closest_E11[pid]:
                            closest_E11[pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"\n  done - scanned {rows_seen:,} rows")

df["closest_K31_84_strictly_before_surgery"] = df["patient_id"].map(closest_K3184)
df["closest_E10_strictly_before_surgery"] = df["patient_id"].map(closest_E10)
df["closest_E11_strictly_before_surgery"] = df["patient_id"].map(closest_E11)

e10_dt = pd.to_datetime(df["closest_E10_strictly_before_surgery"], errors="coerce")
e11_dt = pd.to_datetime(df["closest_E11_strictly_before_surgery"], errors="coerce")
df["closest_E10_or_E11_strictly_before_surgery"] = pd.concat([e10_dt, e11_dt], axis=1).max(axis=1, skipna=True)

# Clean diabetes type label, based on the strictly-before-surgery E10/E11
# columns above (consistent timing framing throughout this script). Since
# this cohort's own inclusion criterion already requires E10 or E11 strictly
# before surgery, this should cover close to 100% of patients either way -
# but using the same timing definition throughout avoids any inconsistency
# between this label and the date columns sitting next to it.
has_e10 = e10_dt.notna()
has_e11 = e11_dt.notna()
diabetes_type = pd.Series("Neither", index=df.index, dtype=object)
diabetes_type[has_e10 & ~has_e11] = "Type 1"
diabetes_type[has_e11 & ~has_e10] = "Type 2"
diabetes_type[has_e10 & has_e11] = "Both"
df["diabetes_type_label"] = diabetes_type

print("\nSANITY CHECK:")
print(f"  patients with a K31.84 code strictly before surgery: {df['closest_K31_84_strictly_before_surgery'].notna().sum():,}/{len(df):,}")
print(f"  patients with an E10 code strictly before surgery:   {df['closest_E10_strictly_before_surgery'].notna().sum():,}/{len(df):,}")
print(f"  patients with an E11 code strictly before surgery:   {df['closest_E11_strictly_before_surgery'].notna().sum():,}/{len(df):,}")
print(f"  diabetes type label breakdown:")
print(diabetes_type.value_counts())

# Hard checks: Script 1's gate and this script's descriptive scan use the
# EXACT same anchor (bariatric_date) and EXACT same window logic against
# the same diagnosis.csv - so if Script 1 found a qualifying occurrence
# exists, this scan is mathematically guaranteed to recover at least that
# same one. A failure here means something genuinely changed between runs,
# not an edge case to shrug off.
if df["closest_K31_84_strictly_before_surgery"].isna().any():
    n_lost = df["closest_K31_84_strictly_before_surgery"].isna().sum()
    raise ValueError(f"{n_lost} gated patients lost their K31.84 date during rescanning - investigate before trusting any output.")

if df["closest_E10_or_E11_strictly_before_surgery"].isna().any():
    n_lost = df["closest_E10_or_E11_strictly_before_surgery"].isna().sum()
    raise ValueError(f"{n_lost} gated patients lost their E10/E11 date during rescanning - investigate before trusting any output.")

print("\nHard consistency checks passed - every gated patient recovered both required dates.")

# Defensive check: nobody should have a K31.84 date ON OR AFTER surgery,
# since the scan filtered to strictly-before only (same-day excluded).
k_dt = pd.to_datetime(df["closest_K31_84_strictly_before_surgery"], errors="coerce")
n_not_strictly_before = (k_dt.notna() & surgery_dt.notna() & (k_dt >= surgery_dt)).sum()
print(f"\n  K31.84 dates on or after surgery (should be 0 - same-day should be excluded): {n_not_strictly_before:,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
