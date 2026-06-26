"""
find_BMI_at_or_before_surgery.py

Scans vitals_signs.csv for baseline BMI for TWO groups simultaneously:
  1. The gastroparesis cohort (from funnel_6_final_cohort.csv)
  2. The comparator pool (from comparator_pool_ready_for_PSM.csv)

For each patient, finds the BMI measurement (LOINC 39156-5) closest to
and ON OR BEFORE their bariatric_date (surgery date). Same-day inclusive
since admission BMI recorded on surgery date is genuinely pre-operative
in a surgical dataset.

Column named BMI_at_or_before_surgery (not "pre-operative BMI") to be
precise - this is the closest BMI on or before the surgery date, which
may include same-day admission measurements.

Running for both groups in one pass saves ~45-60 minutes vs. two scans.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
VITALS_FILE = f"{GCS_BASE}/vitals_signs.csv"

GP_CSV = "cohort_FINAL_analytic.csv"
COMPARATOR_CSV = "comparator_pool_raw.csv"

GP_OUTPUT = "gastroparesis_cohort_BMI_at_or_before_surgery.csv"
COMPARATOR_OUTPUT = "comparator_pool_ready_for_PSM_with_BMI.csv"

BMI_LOINC = "39156-5"

print(">>> SCRIPT VERSION: find_BMI_at_or_before_surgery_v1 <<<")

# Load both groups
gp = pd.read_csv(GP_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "bariatric_date"])
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "bariatric_date"])

print(f"Gastroparesis cohort: {len(gp):,} patients")
print(f"Comparator candidates: {len(comp):,} patients")

# Build surgery date lookups
gp_surgery = dict(zip(gp["patient_id"], pd.to_datetime(gp["bariatric_date"], errors="coerce")))
comp_surgery = dict(zip(comp["patient_id"], pd.to_datetime(comp["bariatric_date"], errors="coerce")))

# Combined set for efficient filtering
all_ids = set(gp_surgery.keys()) | set(comp_surgery.keys())
print(f"Total unique patients to scan: {len(all_ids):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning vitals_signs.csv for BMI (LOINC {BMI_LOINC}) before surgery...")
print(f"File size: ~81 GB - expect 45-60 minutes")

# Store closest pre-op BMI per patient: {pid: (date, value)}
gp_bmi = {}
comp_bmi = {}

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(VITALS_FILE,
                              usecols=["patient_id", "code_system", "code",
                                       "date", "value", "units_of_measure"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()

        bmi = chunk[(chunk["code_system"] == "LOINC") &
                    (chunk["code"] == BMI_LOINC)].copy()
        if not bmi.empty:
            # One-time units sanity check - confirm kg/m2, not % or other
            if chunk_num == 1 and not bmi.empty:
                print(f"  BMI units seen in first chunk: {bmi['units_of_measure'].value_counts(dropna=False).to_dict()}")

            bmi["date"] = pd.to_datetime(bmi["date"], errors="coerce")
            bmi["value"] = pd.to_numeric(bmi["value"], errors="coerce")
            bmi = bmi[bmi["date"].notna() & bmi["value"].notna()]

            # Plausible BMI range filter - catch obvious data errors
            bmi = bmi[(bmi["value"] >= 10) & (bmi["value"] <= 100)]

            if not bmi.empty:
                # Gastroparesis group
                gp_hits = bmi[bmi["patient_id"].isin(gp_surgery)]
                if not gp_hits.empty:
                    gp_hits = gp_hits.copy()
                    gp_hits["surgery_date"] = gp_hits["patient_id"].map(gp_surgery)
                    # Same-day inclusive (<=): admission BMI recorded on surgery
                    # date is genuinely pre-operative in a surgical dataset -
                    # many hospitals record this as part of pre-op paperwork.
                    # Using <= rather than < is documented here for methods section.
                    before = gp_hits[gp_hits["date"] <= gp_hits["surgery_date"]]
                    if not before.empty:
                        for pid, sub in before.groupby("patient_id"):
                            closest_date = sub["date"].max()
                            # Median if multiple BMI readings on same closest date -
                            # more defensible than arbitrary iloc[0]
                            closest_val = sub.loc[sub["date"] == closest_date, "value"].median()
                            if pid not in gp_bmi or closest_date > gp_bmi[pid][0]:
                                gp_bmi[pid] = (closest_date, closest_val)

                # Comparator group
                comp_hits = bmi[bmi["patient_id"].isin(comp_surgery)]
                if not comp_hits.empty:
                    comp_hits = comp_hits.copy()
                    comp_hits["surgery_date"] = comp_hits["patient_id"].map(comp_surgery)
                    # Same-day inclusive - matches GP group logic above
                    before = comp_hits[comp_hits["date"] <= comp_hits["surgery_date"]]
                    if not before.empty:
                        for pid, sub in before.groupby("patient_id"):
                            closest_date = sub["date"].max()
                            closest_val = sub.loc[sub["date"] == closest_date, "value"].median()
                            if pid not in comp_bmi or closest_date > comp_bmi[pid][0]:
                                comp_bmi[pid] = (closest_date, closest_val)

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed, "
              f"GP: {len(gp_bmi):,}/{len(gp_surgery):,} found, "
              f"comparator: {len(comp_bmi):,}/{len(comp_surgery):,} found)")

print(f"\n  done - scanned {rows_seen:,} rows")
elapsed = (time.time() - SCRIPT_START_TIME) / 60
print(f"  Total runtime so far: {elapsed:.1f} minutes")

# --- Write gastroparesis cohort BMI ---
gp["BMI_at_or_before_surgery"] = gp["patient_id"].map(
    lambda p: gp_bmi[p][1] if p in gp_bmi else None)
gp["BMI_date_at_or_before_surgery"] = gp["patient_id"].map(
    lambda p: gp_bmi[p][0] if p in gp_bmi else None)

# --- Write comparator pool BMI ---
comp["BMI_at_or_before_surgery"] = comp["patient_id"].map(
    lambda p: comp_bmi[p][1] if p in comp_bmi else None)
comp["BMI_date_at_or_before_surgery"] = comp["patient_id"].map(
    lambda p: comp_bmi[p][0] if p in comp_bmi else None)

# Missingness comparison - AFTER both columns are assigned
print(f"\nBMI missingness comparison (important for PSM complete-case analysis):")
print(f"  GP missing BMI: {gp['BMI_at_or_before_surgery'].isna().sum():,}/{len(gp):,} "
      f"({100*gp['BMI_at_or_before_surgery'].isna().mean():.1f}%)")
print(f"  Comparator missing BMI: {comp['BMI_at_or_before_surgery'].isna().sum():,}/{len(comp):,} "
      f"({100*comp['BMI_at_or_before_surgery'].isna().mean():.1f}%)")
print("  NOTE: If missingness rates differ substantially between groups,")
print("  complete-case PSM may introduce selection bias - discuss with PI.")

# Range assertions - confirm plausibility filter survived end-to-end
assert gp["BMI_at_or_before_surgery"].dropna().between(10, 100).all(), \
    "GP BMI values outside plausible range (10-100) found in output"
assert comp["BMI_at_or_before_surgery"].dropna().between(10, 100).all(), \
    "Comparator BMI values outside plausible range (10-100) found in output"
print("\nBMI range assertions passed (all values 10-100 kg/m²)")

# Hard check: BMI dates must all be on or before surgery date
gp_bmi_dt = pd.to_datetime(gp["BMI_date_at_or_before_surgery"], errors="coerce")
gp_surg_dt = pd.to_datetime(gp["bariatric_date"], errors="coerce")
n_after = (gp_bmi_dt.notna() & gp_surg_dt.notna() & (gp_bmi_dt > gp_surg_dt)).sum()
print(f"GP BMI dates after surgery (should be 0): {n_after:,}")
n_gp_with_bmi = gp["BMI_at_or_before_surgery"].notna().sum()
print(f"\nGastroparesis cohort BMI coverage: {n_gp_with_bmi:,}/{len(gp):,}")
print(f"  Missing pre-op BMI: {len(gp)-n_gp_with_bmi:,} - these will be excluded from PSM")
print(f"BMI distribution (gastroparesis cohort):")
print(gp["BMI_at_or_before_surgery"].describe())
gp.to_csv(GP_OUTPUT, index=False)
print(f"Wrote {GP_OUTPUT}")

assert comp["patient_id"].is_unique, "Duplicate patient_ids in comparator output"

# Hard check: BMI dates must all be on or before surgery date
comp_bmi_dt = pd.to_datetime(comp["BMI_date_at_or_before_surgery"], errors="coerce")
comp_surg_dt = pd.to_datetime(comp["bariatric_date"], errors="coerce")
n_after_comp = (comp_bmi_dt.notna() & comp_surg_dt.notna() & (comp_bmi_dt > comp_surg_dt)).sum()
print(f"Comparator BMI dates after surgery (should be 0): {n_after_comp:,}")
n_comp_with_bmi = comp["BMI_at_or_before_surgery"].notna().sum()
print(f"\nComparator pool BMI coverage: {n_comp_with_bmi:,}/{len(comp):,}")
print(f"  Missing pre-op BMI: {len(comp)-n_comp_with_bmi:,} - these will be excluded from PSM")
print(f"BMI distribution (comparator pool):")
print(comp["BMI_at_or_before_surgery"].describe())

# NOTE: BMI is a planned PSM covariate. Patients missing pre-op BMI will be
# excluded from PSM in the matching script (dropna on PSM variables).
# If coverage is low (<50%), discuss with Dr. Sujka whether to keep BMI
# as a covariate or drop it to preserve sample size.
comp.to_csv(COMPARATOR_OUTPUT, index=False)
print(f"Wrote {COMPARATOR_OUTPUT}")

print(f"\nTotal runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
