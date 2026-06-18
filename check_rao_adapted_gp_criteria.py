"""
check_rao_adapted_gp_criteria.py

Adapts Rao et al. (Neurogastroenterology & Motility, 2026) gastroparesis
diagnostic criteria to this cohort, using the exact codes from their
Supplementary Table S1. Their criteria, adapted:

  1. K31.84 diagnosed 0 to 90 days AFTER a gastric emptying study
     (widened from their 7-90 days, to include same-day diagnoses -
     deliberate adaptation, see discussion in chat)
  2. At least one typical symptom 90 to 365 days BEFORE diagnosis
  3. An upper endoscopy 30 to 365 days BEFORE diagnosis

DELIBERATELY DROPPED: Rao et al.'s exclusion of patients with a history of
sleeve gastrectomy or gastric bypass. That exclusion would remove this
project's entire population of interest, so it is not applied here. Their
other structural-disease exclusions (gastritis, ulcers, obstruction, etc.)
are also not applied - this script only adds the three POSITIVE criteria
above as new columns, it does not implement Rao's exclusion list at all.
That's a separate decision if it's ever wanted.

CODE NOTE: Rao's table lists early satiety as ICD-10 E68.81, which isn't a
code that exists in standard ICD-10-CM (E68 is "sequelae of overnutrition").
R68.81 ("early satiety," under "other general symptoms") is almost
certainly what was intended. Both are checked here, and the patient counts
matching EACH ONE SEPARATELY are printed so this can be sanity-checked
against the real data once it runs - if E68.81 returns ~0 patients and
R68.81 returns a sensible number, that confirms the typo theory.

Symptom and endoscopy codes have never been scanned for before in this
pipeline - both diagnosis.csv and procedure.csv need a fresh pass,
restricted to cohort patient_ids. The GES timing refinement only needs
arithmetic on first_GES_date, already in the master file - no rescan for
that piece.

Run this AFTER the master file with GES columns exists. Writes a NEW file.
"""

import os
import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"
PROCEDURE_FILE = f"{GCS_BASE}/procedure.csv"

INPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_with_rao_adapted_criteria.csv"

# --- Symptom codes, from Rao et al. Supplementary Table S1 -----------------
NAUSEA_VOMITING_CODES = {"R11", "R11.0", "R11.1", "R11.2", "R11.10", "R11.11"}
EPIGASTRIC_PAIN_CODES = {"R10.13"}
EARLY_SATIETY_CODES = {"E68.81", "R68.81"}  # both checked - see CODE NOTE above
BLOATING_CODES = {"R14.0"}  # proxy for postprandial fullness, per the paper
ALL_SYMPTOM_CODES = NAUSEA_VOMITING_CODES | EPIGASTRIC_PAIN_CODES | EARLY_SATIETY_CODES | BLOATING_CODES

UPPER_ENDOSCOPY_CODES = {"43235", "43239"}

SYMPTOM_WINDOW_MIN_DAYS_BEFORE_DX = 90   # 3 months
SYMPTOM_WINDOW_MAX_DAYS_BEFORE_DX = 365  # 12 months
ENDOSCOPY_WINDOW_MIN_DAYS_BEFORE_DX = 30   # 1 month
ENDOSCOPY_WINDOW_MAX_DAYS_BEFORE_DX = 365  # 12 months
GES_WINDOW_MAX_DAYS_AFTER_GES = 90  # 3 months; min is 0 (same-day allowed, per discussion)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Missing required input file: {INPUT_CSV}")

print(f"Loading {INPUT_CSV} to get the cohort patient list and key dates...")
cohort_df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_df["patient_id"] = cohort_df["patient_id"].astype(str).str.strip()

required_cols = ["first_K31_84_date", "first_GES_date"]
missing_cols = [c for c in required_cols if c not in cohort_df.columns]
if missing_cols:
    raise ValueError(f"{INPUT_CSV} is missing expected columns: {missing_cols}")

dx_date_dt = pd.to_datetime(cohort_df["first_K31_84_date"], errors="coerce")
ges_date_dt = pd.to_datetime(cohort_df["first_GES_date"], errors="coerce")
dx_date_lookup = dict(zip(cohort_df["patient_id"], dx_date_dt))
cohort_ids = set(cohort_df["patient_id"])
print(f"  cohort size: {len(cohort_ids):,} patients ({dx_date_dt.notna().sum():,} with a usable K31.84 date)")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


def scan_for_codes_in_window(gcs_path, code_system_value, code_groups, min_days_before, max_days_before, file_label):
    """
    Scans a GCS file (diagnosis.csv or procedure.csv style) restricted to
    cohort patient_ids, looking for any of several named code groups whose
    date falls within [dx - max_days_before, dx - min_days_before] for that
    patient. Returns {group_name: set(patient_ids)}.
    """
    hits = {name: set() for name in code_groups}
    rows_seen = 0
    chunk_num = 0
    scan_start = time.time()

    for chunk in stream_gcs_csv(gcs_path, usecols=["patient_id", "code_system", "code", "date"]):
        chunk_num += 1
        rows_seen += len(chunk)

        chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
        if not chunk.empty:
            sub = chunk[chunk["code_system"] == code_system_value]
            if not sub.empty:
                sub = sub.copy()
                sub["date"] = pd.to_datetime(sub["date"], format="%Y%m%d", errors="coerce")
                sub["dx_date"] = sub["patient_id"].map(dx_date_lookup)
                days_before_dx = (sub["dx_date"] - sub["date"]).dt.days
                in_window = sub["dx_date"].notna() & sub["date"].notna() & (
                    days_before_dx >= min_days_before
                ) & (days_before_dx <= max_days_before)

                for name, codes in code_groups.items():
                    code_hits = sub[(sub["code"].isin(codes)) & in_window]
                    if not code_hits.empty:
                        hits[name].update(code_hits["patient_id"])

        if chunk_num % 20 == 0:
            elapsed_min = (time.time() - scan_start) / 60
            total_found = len(set().union(*hits.values())) if any(hits.values()) else 0
            print(f"    ...{file_label}: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed, {total_found:,} patients found so far)")

    print(f"  scanned {rows_seen:,} {file_label} rows")
    return hits


print(f"\nScanning diagnosis.csv for symptom codes, {SYMPTOM_WINDOW_MIN_DAYS_BEFORE_DX}-{SYMPTOM_WINDOW_MAX_DAYS_BEFORE_DX} days before dx,")
print("restricted to the cohort's patient_ids...")
symptom_groups = {
    "nausea_vomiting": NAUSEA_VOMITING_CODES,
    "epigastric_pain": EPIGASTRIC_PAIN_CODES,
    "early_satiety_E68_81": {"E68.81"},
    "early_satiety_R68_81": {"R68.81"},
    "bloating": BLOATING_CODES,
}
symptom_hits = scan_for_codes_in_window(
    DIAGNOSIS_FILE, "ICD-10-CM", symptom_groups,
    SYMPTOM_WINDOW_MIN_DAYS_BEFORE_DX, SYMPTOM_WINDOW_MAX_DAYS_BEFORE_DX, "diagnosis.csv"
)

print(f"\nScanning procedure.csv for upper endoscopy codes, {ENDOSCOPY_WINDOW_MIN_DAYS_BEFORE_DX}-{ENDOSCOPY_WINDOW_MAX_DAYS_BEFORE_DX} days before dx,")
print("restricted to the cohort's patient_ids...")
endoscopy_groups = {"upper_endoscopy": UPPER_ENDOSCOPY_CODES}
endoscopy_hits = scan_for_codes_in_window(
    PROCEDURE_FILE, "CPT", endoscopy_groups,
    ENDOSCOPY_WINDOW_MIN_DAYS_BEFORE_DX, ENDOSCOPY_WINDOW_MAX_DAYS_BEFORE_DX, "procedure.csv"
)

# --- Assemble columns -------------------------------------------------------
any_early_satiety = symptom_hits["early_satiety_E68_81"] | symptom_hits["early_satiety_R68_81"]
any_symptom = (
    symptom_hits["nausea_vomiting"]
    | symptom_hits["epigastric_pain"]
    | any_early_satiety
    | symptom_hits["bloating"]
)

cohort_df["has_nausea_vomiting_3to12mo_before_dx"] = cohort_df["patient_id"].isin(symptom_hits["nausea_vomiting"])
cohort_df["has_epigastric_pain_3to12mo_before_dx"] = cohort_df["patient_id"].isin(symptom_hits["epigastric_pain"])
cohort_df["has_early_satiety_E68_81_3to12mo_before_dx"] = cohort_df["patient_id"].isin(symptom_hits["early_satiety_E68_81"])
cohort_df["has_early_satiety_R68_81_3to12mo_before_dx"] = cohort_df["patient_id"].isin(symptom_hits["early_satiety_R68_81"])
cohort_df["has_bloating_3to12mo_before_dx"] = cohort_df["patient_id"].isin(symptom_hits["bloating"])
cohort_df["has_any_typical_symptom_3to12mo_before_dx"] = cohort_df["patient_id"].isin(any_symptom)

cohort_df["has_upper_endoscopy_1to12mo_before_dx"] = cohort_df["patient_id"].isin(endoscopy_hits["upper_endoscopy"])

# GES timing refinement - pure arithmetic, no rescan needed.
days_dx_after_ges = (dx_date_dt - ges_date_dt).dt.days
cohort_df["ges_within_90d_before_dx"] = (
    ges_date_dt.notna() & dx_date_dt.notna() & (days_dx_after_ges >= 0) & (days_dx_after_ges <= GES_WINDOW_MAX_DAYS_AFTER_GES)
)

cohort_df["meets_rao_adapted_gp_criteria"] = (
    cohort_df["ges_within_90d_before_dx"]
    & cohort_df["has_any_typical_symptom_3to12mo_before_dx"]
    & cohort_df["has_upper_endoscopy_1to12mo_before_dx"]
)

print("\nSANITY CHECK:")
in_period = cohort_df["in_study_period"]
print(f"\nOf the {in_period.sum():,} in-study-period K31.84 patients:")
print(f"  GES within 0-{GES_WINDOW_MAX_DAYS_AFTER_GES} days before dx: {(in_period & cohort_df['ges_within_90d_before_dx']).sum():,}")
print(f"  nausea/vomiting in symptom window: {(in_period & cohort_df['has_nausea_vomiting_3to12mo_before_dx']).sum():,}")
print(f"  epigastric pain in symptom window: {(in_period & cohort_df['has_epigastric_pain_3to12mo_before_dx']).sum():,}")
print(f"  early satiety (E68.81) in symptom window: {(in_period & cohort_df['has_early_satiety_E68_81_3to12mo_before_dx']).sum():,}  <- watch this one, likely a typo in the source table")
print(f"  early satiety (R68.81) in symptom window: {(in_period & cohort_df['has_early_satiety_R68_81_3to12mo_before_dx']).sum():,}")
print(f"  bloating in symptom window: {(in_period & cohort_df['has_bloating_3to12mo_before_dx']).sum():,}")
print(f"  ANY typical symptom in symptom window: {(in_period & cohort_df['has_any_typical_symptom_3to12mo_before_dx']).sum():,}")
print(f"  upper endoscopy in endoscopy window: {(in_period & cohort_df['has_upper_endoscopy_1to12mo_before_dx']).sum():,}")
print(f"  MEETS ALL THREE Rao-adapted criteria: {(in_period & cohort_df['meets_rao_adapted_gp_criteria']).sum():,}")

cohort_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with the columns above added.")
print(f"(Original {INPUT_CSV} left untouched - use the new file as input to the next step.)")

total_elapsed_min = (time.time() - SCRIPT_START_TIME) / 60
print(f"\nTotal script runtime: {total_elapsed_min:.1f} minutes")
