"""
build_gastroparesis_prokinetic_cohort.py

Step: identify gastroparesis-diagnosed patients (any year, so we can see
how many legacy 536.3 patients exist for documentation) and report, for
each, the first/last date of each prokinetic drug (metoclopramide /
erythromycin / domperidone / prucalopride) found anywhere in their record.

IMPORTANT: prokinetic drug evidence is descriptive only here, NOT a hard
inclusion criterion. The cohort is defined purely by K31.84 on/after the
study start date. Whether to require drug evidence on top of that is a
separate decision to make once you've seen the actual counts.

Cohort logic (confirmed with Dr. Sujka, June 2026):
  - Main analytic cohort = first ICD-10-CM K31.84 diagnosis on/after
    2015-10-01 (US ICD-10-CM go-live date)
  - ICD-9-CM 536.3 (legacy gastroparesis code) is tracked separately for
    documentation but does NOT count toward the main cohort

RxNorm ingredient codes confirmed directly from this dataset's
standardized_terminology.csv:
    6915      metoclopramide
    4053      erythromycin    (NOTE: also covers topical/ophthalmic use -
                                code alone can't isolate prokinetic intent)
    3626      domperidone
    2107310   prucalopride

All dates are parsed into real datetime objects (not compared as raw
strings) to avoid silent failures if the data ever mixes date formats.

Run on the VM with: python3 build_gastroparesis_prokinetic_cohort.py
Requires: gsutil already authenticated, pandas installed
  (pip install pandas --break-system-packages if needed)
"""

import subprocess
import time
import pandas as pd

SCRIPT_START_TIME = time.time()

# ---------------------------------------------------------------------------
# CONFIG - edit these if paths or codes change
# ---------------------------------------------------------------------------

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"
MED_INGREDIENT_FILE = f"{GCS_BASE}/medication_ingredient.csv"
MED_DRUG_FILE = f"{GCS_BASE}/medication_drug.csv"

OUTPUT_CSV = "gastroparesis_prokinetic_cohort.csv"
ERYTHROMYCIN_DETAIL_CSV = "erythromycin_drug_detail.csv"

ERYTHROMYCIN_CODE = "4053"  # the one drug with form/route ambiguity worth detailing

STUDY_START_DATE = pd.Timestamp("2015-10-01")  # ICD-10-CM go-live date

GASTROPARESIS_DX_CODES = {
    "ICD-10-CM": "K31.84",
    "ICD-9-CM": "536.3",
}

PROKINETIC_RXNORM_CODES = {
    "6915": "metoclopramide",
    "4053": "erythromycin",
    "3626": "domperidone",
    "2107310": "prucalopride",
}

CHUNK_SIZE = 500_000  # rows per chunk; lower this if the VM is memory-tight

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stream_gcs_csv(gcs_path, usecols, chunksize=CHUNK_SIZE):
    """
    Stream a CSV directly from GCS via `gsutil cat`, never writing the full
    file to local disk. Yields pandas DataFrame chunks.
    """
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    reader = pd.read_csv(
        proc.stdout,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        yield chunk
    proc.stdout.close()
    proc.wait()


def merge_min(running, new_vals):
    for pid, val in new_vals.items():
        if pid not in running or val < running[pid]:
            running[pid] = val


def merge_max(running, new_vals):
    for pid, val in new_vals.items():
        if pid not in running or val > running[pid]:
            running[pid] = val


def parse_dates(series, label=""):
    """
    Convert a YYYYMMDD string column to real datetime values. Anything that
    doesn't match that format becomes NaT (and gets counted/printed) rather
    than silently comparing wrong via raw string comparison.
    """
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    bad = parsed.isna() & series.notna()
    if bad.any():
        print(f"  WARNING: {bad.sum():,} unparseable dates in {label or 'a date column'} (set to NaT, excluded from all comparisons)")
    return parsed


# ---------------------------------------------------------------------------
# Step 1: first-diagnosis-date index for K31.84 and legacy 536.3
# ---------------------------------------------------------------------------

print("Scanning diagnosis.csv for gastroparesis codes (K31.84, 536.3)...")

first_k3184 = {}   # patient_id -> earliest datetime (or NaT/missing)
last_k3184 = {}    # patient_id -> latest datetime (or NaT/missing)
first_5363 = {}    # patient_id -> earliest datetime (or NaT/missing)
k3184_encounters = {}  # patient_id -> set of distinct, non-null encounter_ids with a K31.84 dx
k3184_missing_encounter_rows = 0

rows_seen = 0
chunk_num = 0
diag_start = time.time()
for chunk in stream_gcs_csv(
    DIAGNOSIS_FILE,
    usecols=["patient_id", "encounter_id", "code_system", "code", "date"],
):
    chunk_num += 1
    rows_seen += len(chunk)
    chunk["date"] = parse_dates(chunk["date"], label="diagnosis.csv date")

    k_mask = (chunk["code_system"] == "ICD-10-CM") & (
        chunk["code"] == GASTROPARESIS_DX_CODES["ICD-10-CM"]
    )
    if k_mask.any():
        k_sub = chunk.loc[k_mask]
        k_min = k_sub.groupby("patient_id")["date"].min().to_dict()
        k_max = k_sub.groupby("patient_id")["date"].max().to_dict()
        merge_min(first_k3184, k_min)
        merge_max(last_k3184, k_max)

        k3184_missing_encounter_rows += k_sub["encounter_id"].isna().sum()

        for pid, enc_id in zip(k_sub["patient_id"], k_sub["encounter_id"]):
            if pd.notna(enc_id):
                k3184_encounters.setdefault(pid, set()).add(enc_id)

    legacy_mask = (chunk["code_system"] == "ICD-9-CM") & (
        chunk["code"] == GASTROPARESIS_DX_CODES["ICD-9-CM"]
    )
    if legacy_mask.any():
        l_min = chunk.loc[legacy_mask].groupby("patient_id")["date"].min().to_dict()
        merge_min(first_5363, l_min)

    if chunk_num % 20 == 0:
        elapsed_min = (time.time() - diag_start) / 60
        print(f"    ...diagnosis.csv: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed, {len(first_k3184):,} K31.84 patients found)")

k3184_encounter_count = {pid: len(encs) for pid, encs in k3184_encounters.items()}
patients_with_encounter_data = len(k3184_encounter_count)
patients_without_encounter_data = len(first_k3184) - patients_with_encounter_data
multi_encounter_count = sum(1 for c in k3184_encounter_count.values() if c >= 2)
single_encounter_count = patients_with_encounter_data - multi_encounter_count

print(f"  scanned {rows_seen:,} diagnosis rows")
print(f"  patients with K31.84 (any year): {len(first_k3184):,}")
print(f"  patients with legacy 536.3:      {len(first_5363):,}")
print(f"  K31.84 rows missing encounter_id: {k3184_missing_encounter_rows:,}")
if k3184_missing_encounter_rows > 0:
    print("  NOTE: rows with missing encounter_id are excluded from the distinct-encounter")
    print("  count above, so num_K31_84_distinct_encounters may understate the true count")
    print("  for patients whose K31.84 rows lack an encounter_id.")

all_gastroparesis_patients = set(first_k3184) | set(first_5363)
print(f"  total gastroparesis patients (either code, any year): {len(all_gastroparesis_patients):,}")
print(f"  K31.84 patients with >=1 row containing a valid encounter_id: {patients_with_encounter_data:,}")
print(f"    of those: {multi_encounter_count:,} have >=2 distinct encounters, {single_encounter_count:,} have exactly 1")
print(f"  K31.84 patients with NO valid encounter_id on any row: {patients_without_encounter_data:,}")

# ---------------------------------------------------------------------------
# Step 2: first AND last recorded date per patient per prokinetic drug
#   (first date = earliest ever recorded; last date = most recent recorded;
#   both are purely descriptive here, not used to gate inclusion)
# ---------------------------------------------------------------------------

print("\nScanning medication_ingredient.csv for prokinetic drug codes...")

drug_first_date = {code: {} for code in PROKINETIC_RXNORM_CODES}
drug_last_date = {code: {} for code in PROKINETIC_RXNORM_CODES}
erythromycin_ingredient_records = []  # one entry per erythromycin hit, for drug-detail lookup

rows_seen = 0
chunk_num = 0
med_ing_start = time.time()
for chunk in stream_gcs_csv(
    MED_INGREDIENT_FILE,
    usecols=["patient_id", "unique_id", "code_system", "code", "start_date"],
):
    chunk_num += 1
    rows_seen += len(chunk)
    chunk["start_date"] = parse_dates(chunk["start_date"], label="medication_ingredient.csv start_date")

    rx_chunk = chunk[chunk["code_system"] == "RxNorm"]
    if rx_chunk.empty:
        if chunk_num % 20 == 0:
            elapsed_min = (time.time() - med_ing_start) / 60
            print(f"    ...medication_ingredient.csv: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed)")
        continue

    for code in PROKINETIC_RXNORM_CODES:
        code_mask = rx_chunk["code"] == code
        if code_mask.any():
            sub = rx_chunk.loc[code_mask]
            min_dates = sub.groupby("patient_id")["start_date"].min().to_dict()
            max_dates = sub.groupby("patient_id")["start_date"].max().to_dict()
            merge_min(drug_first_date[code], min_dates)
            merge_max(drug_last_date[code], max_dates)

            if code == ERYTHROMYCIN_CODE:
                erythromycin_ingredient_records.extend(
                    sub[["patient_id", "unique_id", "start_date"]].to_dict("records")
                )

    if chunk_num % 20 == 0:
        elapsed_min = (time.time() - med_ing_start) / 60
        print(f"    ...medication_ingredient.csv: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed)")

print(f"  scanned {rows_seen:,} medication_ingredient rows")
for code, name in PROKINETIC_RXNORM_CODES.items():
    print(f"  patients with {name} (RxCUI {code}) anywhere in record: {len(drug_last_date[code]):,}")

# ---------------------------------------------------------------------------
# Step 2.5: pull route / brand / strength / quantity / days_supply for the
#   erythromycin records, by joining on unique_id into medication_drug.csv
#   (unique_id links a single medication event across the ingredient-level
#   and product-level files - this is the assumption being made here, and
#   the match-rate printed below is a sanity check on it)
# ---------------------------------------------------------------------------

print(f"\nFound {len(erythromycin_ingredient_records):,} erythromycin ingredient records to cross-reference for drug detail.")

erythromycin_unique_ids = {r["unique_id"] for r in erythromycin_ingredient_records}
drug_detail_lookup = {}  # unique_id -> {route, brand, strength, quantity_dispensed, days_supply}

print("Scanning medication_drug.csv for matching product-level detail...")

rows_seen = 0
chunk_num = 0
med_drug_start = time.time()
for chunk in stream_gcs_csv(
    MED_DRUG_FILE,
    usecols=["unique_id", "route", "brand", "strength", "quantity_dispensed", "days_supply"],
):
    chunk_num += 1
    rows_seen += len(chunk)
    match_mask = chunk["unique_id"].isin(erythromycin_unique_ids)
    if match_mask.any():
        for rec in chunk.loc[match_mask].to_dict("records"):
            drug_detail_lookup[rec["unique_id"]] = rec

    if chunk_num % 20 == 0:
        elapsed_min = (time.time() - med_drug_start) / 60
        print(f"    ...medication_drug.csv: {rows_seen:,} rows processed so far ({elapsed_min:.1f} min elapsed)")

print(f"  scanned {rows_seen:,} medication_drug rows")
print(f"  matched product-level detail for {len(drug_detail_lookup):,} of {len(erythromycin_unique_ids):,} erythromycin records")
if erythromycin_unique_ids and len(drug_detail_lookup) < 0.9 * len(erythromycin_unique_ids):
    print("  WARNING: match rate below 90% - double check that unique_id is really a shared join key between these two files before trusting the detail output.")


def is_oral_route(route):
    return isinstance(route, str) and "oral" in route.lower()


erythromycin_detail_rows = []
oral_evidence_patients = set()

for rec in erythromycin_ingredient_records:
    pid = rec["patient_id"]
    dx_date = first_k3184.get(pid)
    on_or_after_dx = (
        pd.notna(dx_date) and pd.notna(rec["start_date"]) and rec["start_date"] >= dx_date
    )
    detail = drug_detail_lookup.get(rec["unique_id"], {})
    route = detail.get("route")

    erythromycin_detail_rows.append({
        "patient_id": pid,
        "unique_id": rec["unique_id"],
        "start_date": rec["start_date"],
        "on_or_after_first_K31_84_dx": on_or_after_dx,
        "route": route,
        "brand": detail.get("brand"),
        "strength": detail.get("strength"),
        "quantity_dispensed": detail.get("quantity_dispensed"),
        "days_supply": detail.get("days_supply"),
    })

    if on_or_after_dx and is_oral_route(route):
        oral_evidence_patients.add(pid)

erythromycin_detail_df = pd.DataFrame(erythromycin_detail_rows)
erythromycin_detail_df.to_csv(ERYTHROMYCIN_DETAIL_CSV, index=False)
print(f"Wrote {len(erythromycin_detail_df):,} erythromycin detail rows to {ERYTHROMYCIN_DETAIL_CSV}")
print("  Review this file manually for dose/duration patterns (e.g. short course = likely")
print("  antibiotic use, extended or recurring low-dose = more consistent with prokinetic use).")

# ---------------------------------------------------------------------------
# Step 3: assemble patient-level CSV
# ---------------------------------------------------------------------------

print("\nAssembling patient-level output...")

records = []
for pid in sorted(all_gastroparesis_patients):
    k3184_date = first_k3184.get(pid)
    legacy_date = first_5363.get(pid)

    in_study_period = pd.notna(k3184_date) and k3184_date >= STUDY_START_DATE

    row = {
        "patient_id": pid,
        "first_K31_84_date": k3184_date,
        "last_K31_84_date": last_k3184.get(pid),
        "days_between_first_and_last_K31_84": (
            (last_k3184.get(pid) - k3184_date).days
            if pd.notna(k3184_date) and pd.notna(last_k3184.get(pid))
            else None
        ),
        "num_K31_84_distinct_encounters": k3184_encounter_count.get(pid, 0),
        "first_536_3_date": legacy_date,
        "in_study_period": in_study_period,
    }

    any_prokinetic_any_time = False
    any_prokinetic_ever_after_dx = False
    num_any_time = 0
    num_ever_after_dx = 0

    for code, name in PROKINETIC_RXNORM_CODES.items():
        first_date = drug_first_date[code].get(pid)
        last_date = drug_last_date[code].get(pid)
        # NOTE: this only proves at least one record exists on/after dx
        # (max date >= dx date implies SOME exposure after dx) - it does
        # NOT mean every record for this drug falls after dx, hence the
        # "ever" in the name rather than something implying continuity.
        ever_after_dx = (
            pd.notna(k3184_date) and pd.notna(last_date) and last_date >= k3184_date
        )

        row[f"{name}_first_date"] = first_date
        row[f"{name}_last_date"] = last_date
        row[f"{name}_ever_after_dx"] = ever_after_dx

        if pd.notna(first_date):
            any_prokinetic_any_time = True
            num_any_time += 1
        if ever_after_dx:
            any_prokinetic_ever_after_dx = True
            num_ever_after_dx += 1

    row["any_prokinetic_any_time"] = any_prokinetic_any_time
    row["any_prokinetic_ever_after_dx"] = any_prokinetic_ever_after_dx
    row["num_prokinetic_drugs_any_time"] = num_any_time
    row["num_prokinetic_drugs_ever_after_dx"] = num_ever_after_dx
    row["erythromycin_oral_evidence_ever_after_dx"] = pid in oral_evidence_patients

    records.append(row)

cohort_df = pd.DataFrame(records)
cohort_df.to_csv(OUTPUT_CSV, index=False)

# ---------------------------------------------------------------------------
# Step 4: summary counts
# ---------------------------------------------------------------------------

print(f"\nWrote {len(cohort_df):,} patients to {OUTPUT_CSV}")
print("\n--- Summary (descriptive only - no drug-based inclusion/exclusion applied) ---")
print(f"Total gastroparesis patients (K31.84 or 536.3, any year): {len(cohort_df):,}")
only_legacy = (
    cohort_df["first_K31_84_date"].isna() & cohort_df["first_536_3_date"].notna()
).sum()
print(f"  ...of which only 536.3, no K31.84 ever:  {only_legacy:,}")
print(f"  ...of which have K31.84:                  {cohort_df['first_K31_84_date'].notna().sum():,}")
print(f"In study period (K31.84 on/after {STUDY_START_DATE.date()}): {cohort_df['in_study_period'].sum():,}")

in_period = cohort_df["in_study_period"]
print(f"\nOf the {in_period.sum():,} in-study-period K31.84 patients:")
zero_encounter_in_period = (in_period & (cohort_df["num_K31_84_distinct_encounters"] == 0)).sum()
single_encounter_in_period = (in_period & (cohort_df["num_K31_84_distinct_encounters"] == 1)).sum()
multi_encounter_in_period = (in_period & (cohort_df["num_K31_84_distinct_encounters"] >= 2)).sum()
print(f"  >=2 distinct K31.84 encounters:  {multi_encounter_in_period:,}")
print(f"  exactly 1 K31.84 encounter:      {single_encounter_in_period:,}")
print(f"  0 valid encounter_id (data gap): {zero_encounter_in_period:,}")

recurring_mask = cohort_df["days_between_first_and_last_K31_84"].fillna(0) > 0
print(f"  K31.84 spanning >0 days between first and last diagnosis: {(in_period & recurring_mask).sum():,}")

# Encounter count and date span are complementary, not interchangeable - a patient
# can have >=2 distinct encounter_ids on the SAME calendar date (e.g. separate
# inpatient/outpatient billing same day), which is a different signal than
# encounters spread across months or years.
multi_enc_mask = in_period & (cohort_df["num_K31_84_distinct_encounters"] >= 2)
multi_enc_same_day = multi_enc_mask & (cohort_df["days_between_first_and_last_K31_84"].fillna(0) == 0)
multi_enc_spanning = multi_enc_mask & (cohort_df["days_between_first_and_last_K31_84"].fillna(0) > 0)
print(f"  of the >=2-encounter group: {multi_enc_spanning.sum():,} span >0 days, {multi_enc_same_day.sum():,} are same-day despite multiple encounter_ids")

for name in PROKINETIC_RXNORM_CODES.values():
    any_time_count = cohort_df.loc[in_period, f"{name}_first_date"].notna().sum()
    after_dx_count = cohort_df.loc[in_period, f"{name}_ever_after_dx"].sum()
    print(f"  {name}: {any_time_count:,} have it anywhere in record ({after_dx_count:,} of those ever after dx)")

print(f"  any of the 4 drugs, anywhere in record: {cohort_df.loc[in_period, 'any_prokinetic_any_time'].sum():,}")
print(f"  any of the 4 drugs, ever after dx:      {cohort_df.loc[in_period, 'any_prokinetic_ever_after_dx'].sum():,}")
print(f"  none of the 4 drugs at all:              {(in_period & ~cohort_df['any_prokinetic_any_time']).sum():,}")

erythromycin_only_mask = (
    in_period
    & cohort_df["erythromycin_ever_after_dx"]
    & ~cohort_df["metoclopramide_ever_after_dx"]
    & ~cohort_df["domperidone_ever_after_dx"]
    & ~cohort_df["prucalopride_ever_after_dx"]
)
print(f"\n  of the ever-after-dx group, erythromycin is the ONLY drug for: {erythromycin_only_mask.sum():,}")
print(f"  of those, how many have oral-route evidence ever after dx: {(erythromycin_only_mask & cohort_df['erythromycin_oral_evidence_ever_after_dx']).sum():,}")

total_elapsed_min = (time.time() - SCRIPT_START_TIME) / 60
print(f"\nTotal script runtime: {total_elapsed_min:.1f} minutes")
