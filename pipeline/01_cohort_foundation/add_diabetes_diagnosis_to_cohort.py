"""
add_diabetes_diagnosis_to_cohort.py

Adds diabetes diagnosis evidence (ICD-10-CM E08-E13, the standard "Diabetes
mellitus" code block) to the gastroparesis cohort.

SCOPE NOTE: only ICD-10-CM E08-E13 is checked here, per request - this does
NOT include the legacy ICD-9-CM 250.xx diabetes codes. A patient diagnosed
with diabetes only before Oct 2015 (ICD-10 go-live) under 250.xx would show
has_diabetes_dx = False here even if they're truly diabetic - same kind of
boundary issue as the K31.84/536.3 split earlier. Flag to Dr. Sujka if this
matters for your population (likely a small effect, since most patients
with diabetes will also have it re-coded under E08-E13 at some point in a
typical ~7-year HCO record).

DESIGN NOTE: this is a standalone comorbidity flag, NOT anchored to the
gastroparesis diagnosis date or to any "concurrency" window. Sadda et al.'s
5-year concurrency rule (the later of diabetes-dx and gastroparesis-dx must
fall within 5 years before the surgical index date) genuinely doesn't apply
yet at this stage of the pipeline, since there's no surgery/index date in
play here - this script only builds the base gastroparesis population. Once
this cohort gets joined with bariatric surgery records (which already carry
a bariatric_date field from the matched-pairs work), THAT'S the point to
apply Sadda's actual rule: require max(first_diabetes_dx_date,
first_K31_84_date) to fall within 5 years before bariatric_date. For now,
this just records whether and when each patient has a diabetes dx code on
record - purely descriptive, no inclusion/exclusion criteria applied.

Restricts the diagnosis.csv scan to patients already in the cohort (rather
than tracking diabetes for the entire ~2M-patient dataset), since diabetes
is far more prevalent than gastroparesis and there's no reason to carry
data for patients outside the cohort.

Run this AFTER check_gastric_emptying_study.py. Writes a NEW file rather
than overwriting the GES output, so re-running earlier steps doesn't wipe
out these columns.
"""

import os
import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"
COHORT_CSV = "gastroparesis_prokinetic_cohort_with_GES.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_and_diabetes.csv"

# ICD-10-CM "Diabetes mellitus" block. E12 is not actually an assigned
# category in ICD-10-CM, but included defensively since TriNetX does not
# clean or correct nonstandard HCO-supplied codes (per their FAQ).
DIABETES_ICD10_PREFIXES = ("E08", "E09", "E10", "E11", "E12", "E13")

if not os.path.exists(COHORT_CSV):
    raise FileNotFoundError(
        f"Missing required input file: {COHORT_CSV} - run check_gastric_emptying_study.py first"
    )

print(f"Loading {COHORT_CSV} to get the cohort patient list...")
cohort_df = pd.read_csv(COHORT_CSV, dtype={"patient_id": str})

if "in_study_period" not in cohort_df.columns:
    raise ValueError(f"{COHORT_CSV} is missing the expected 'in_study_period' column")

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


print(f"\nScanning diagnosis.csv for diabetes codes ({', '.join(DIABETES_ICD10_PREFIXES)}),")
print("restricted to the cohort's patient_ids...")

diabetes_first_date = {}
diabetes_codes_seen = {}

rows_seen = 0
chunk_num = 0
diag_start = time.time()
for chunk in stream_gcs_csv(DIAGNOSIS_FILE, usecols=["patient_id", "code_system", "code", "date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if chunk.empty:
        continue

    mask = (chunk["code_system"] == "ICD-10-CM") & (chunk["code"].str[:3].isin(DIABETES_ICD10_PREFIXES))
    if mask.any():
        hits = chunk.loc[mask, ["patient_id", "code", "date"]].copy()
        hits["date"] = pd.to_datetime(hits["date"], format="%Y%m%d", errors="coerce")

        for pid, sub in hits.groupby("patient_id"):
            diabetes_codes_seen.setdefault(pid, set()).update(sub["code"])
            min_date = sub["date"].min()
            if pd.notna(min_date) and (
                pid not in diabetes_first_date
                or pd.isna(diabetes_first_date[pid])
                or min_date < diabetes_first_date[pid]
            ):
                diabetes_first_date[pid] = min_date

    if chunk_num % 20 == 0:
        elapsed_min = (time.time() - diag_start) / 60
        print(
            f"    ...diagnosis.csv: {rows_seen:,} rows processed so far "
            f"({elapsed_min:.1f} min elapsed, {len(diabetes_first_date):,} cohort diabetes patients found)"
        )

print(f"\n  scanned {rows_seen:,} diagnosis rows")
print(f"  cohort patients with a diabetes dx code (any year): {len(diabetes_first_date):,}")

diabetes_patients = set(diabetes_first_date)
cohort_df["has_diabetes_dx"] = cohort_df["patient_id"].isin(diabetes_patients)
cohort_df["first_diabetes_dx_date"] = cohort_df["patient_id"].map(diabetes_first_date)

diabetes_code_types = {pid: ",".join(sorted(codes)) for pid, codes in diabetes_codes_seen.items()}
cohort_df["diabetes_icd10_codes_seen"] = cohort_df["patient_id"].map(diabetes_code_types)


def has_prefix(codes, prefixes):
    return any(c.startswith(p) for c in codes for p in prefixes)


cohort_df["has_type1_diabetes_code"] = cohort_df["patient_id"].map(
    lambda pid: has_prefix(diabetes_codes_seen.get(pid, set()), ("E10",))
)
cohort_df["has_type2_diabetes_code"] = cohort_df["patient_id"].map(
    lambda pid: has_prefix(diabetes_codes_seen.get(pid, set()), ("E11",))
)
cohort_df["has_other_diabetes_code"] = cohort_df["patient_id"].map(
    lambda pid: has_prefix(diabetes_codes_seen.get(pid, set()), ("E08", "E09", "E12", "E13"))
)
cohort_df["has_ambiguous_diabetes_type"] = (
    cohort_df["has_type1_diabetes_code"] & cohort_df["has_type2_diabetes_code"]
)

print("\nSANITY CHECK:")
print(f"  cohort patients with a diabetes dx: {len(diabetes_first_date):,}")
print(f"  cohort size (all gastroparesis patients, any year): {len(cohort_df):,}")
print(f"  overlap rate (has_diabetes_dx, full cohort): {cohort_df['has_diabetes_dx'].mean() * 100:.2f}%")

in_period = cohort_df["in_study_period"]
print(f"\nOf the {in_period.sum():,} in-study-period K31.84 patients:")
print(f"  have a diabetes dx on record (any year): {(in_period & cohort_df['has_diabetes_dx']).sum():,}")
print(f"  do NOT have a diabetes dx on record:     {(in_period & ~cohort_df['has_diabetes_dx']).sum():,}")
print(f"  Type 1 (E10) code present:               {(in_period & cohort_df['has_type1_diabetes_code']).sum():,}")
print(f"  Type 2 (E11) code present:                {(in_period & cohort_df['has_type2_diabetes_code']).sum():,}")
print(f"  Other (E08/E09/E12/E13) code present:    {(in_period & cohort_df['has_other_diabetes_code']).sum():,}")
print(
    f"  Ambiguous (both Type 1 AND Type 2 codes present): "
    f"{(in_period & cohort_df['has_ambiguous_diabetes_type']).sum():,}"
)

cohort_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with has_diabetes_dx, first_diabetes_dx_date,")
print("diabetes_icd10_codes_seen, has_type1_diabetes_code, has_type2_diabetes_code,")
print("has_other_diabetes_code, and has_ambiguous_diabetes_type columns added.")
print(f"(Original {COHORT_CSV} left untouched - use the new file as input to the next step.)")
print("\nThis is descriptive only - decide with Dr. Sujka whether/how to use diabetes status")
print("(e.g. as a stratification variable, an inclusion criterion, or purely descriptive).")

total_elapsed_min = (time.time() - SCRIPT_START_TIME) / 60
print(f"\nTotal script runtime: {total_elapsed_min:.1f} minutes")
