"""
collect_medications_all.py

Single medication_ingredient.csv scan collecting ALL medication-related
outcomes for both PSM versions simultaneously:

1. PROKINETICS in both groups
   (primary use: characterize comparator contamination and compare
   prokinetic exposure between gastroparesis and comparator patients)
2. DIABETES MEDICATIONS for all matched patients:
   - Insulin (various)
   - Metformin
   - GLP-1 agonists (semaglutide, liraglutide, dulaglutide, exenatide,
     tirzepatide)
   - SGLT2 inhibitors (empagliflozin, dapagliflozin, canagliflozin)
   Used for: MEDICATION COMPONENT of ADA remission criteria.
   NOTE: ADA remission requires A1c <6.5% AND off glucose-lowering
   medications for ≥3 months. This script provides the medication data
   only. Combine with a1c_trajectory files to compute full remission.
   Also used for: pre/post medication comparison across timepoints.

Windows (same as A1c and ED):
  - Pre-op: 365 days before surgery to day -1 (active medication window)
  - Year 1: days 31-365
  - Year 2: days 366-730
  - Year 3: days 731-1095
  - Year 4: days 1096-1460
  - Year 5: days 1461-1825

Note: rx_record_count counts medication RECORDS in TriNetX, not actual
prescriptions - refills and renewals may appear as separate rows.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MED_FILE = f"{GCS_BASE}/medication_ingredient.csv"

MASTER_CSV = "master_cohort_FINAL_1118.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
PAIRS_BMI_CSV = "psm_matched_pairs.csv"
PAIRS_NO_BMI_CSV = "psm_matched_pairs_no_BMI.csv"

# Prokinetics
PROKINETIC_CODES = {
    "6915": "metoclopramide",
    "4053": "erythromycin",
    "3626": "domperidone",
    "2107310": "prucalopride",
}

# Diabetes medications
INSULIN_CODES = {
    "5856": "insulin", "51428": "insulin_glargine", "86009": "insulin_lispro",
    "274783": "insulin_aspart", "253182": "insulin_detemir",
    "865098": "insulin_degludec", "1373463": "insulin_glulisine",
}
METFORMIN_CODES = {"6809": "metformin"}
GLP1_CODES = {
    "2200644": "semaglutide", "475968": "liraglutide",
    "1534763": "dulaglutide", "60548": "exenatide",
    "2397641": "tirzepatide",
}
SGLT2_CODES = {
    "1488564": "empagliflozin", "1373458": "dapagliflozin",
    "1649380": "canagliflozin",
}

ALL_DIABETES_CODES = {**INSULIN_CODES, **METFORMIN_CODES, **GLP1_CODES, **SGLT2_CODES}
ALL_CODES = {**PROKINETIC_CODES, **ALL_DIABETES_CODES}

WINDOWS = {
    "pre_op":  (-365, -1),   # within 1 year before surgery - "active medication"
    "year_1":  (31, 365),    # consistent with A1c and ED windows
    "year_2":  (366, 730),
    "year_3":  (731, 1095),
    "year_4":  (1096, 1460),
    "year_5":  (1461, 1825),
}

print(">>> SCRIPT VERSION: collect_medications_all_v1 <<<")

# --- Load surgery dates ---
master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "bariatric_date"])
master_dates = dict(zip(master["patient_id"],
                         pd.to_datetime(master["bariatric_date"], errors="coerce")))
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "bariatric_date"])
comp_dates = dict(zip(comp["patient_id"],
                       pd.to_datetime(comp["bariatric_date"], errors="coerce")))
surgery_dates = {**master_dates, **comp_dates}

# --- Load both matched pair sets ---
def load_pair_ids(csv_path):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    gp_col = [c for c in df.columns if "patient_id" in c and "_gp" in c][0]
    comp_col = [c for c in df.columns if "patient_id" in c and "_comp" in c][0]
    return set(df[gp_col].dropna()), set(df[comp_col].dropna())

gp_bmi, comp_bmi = load_pair_ids(PAIRS_BMI_CSV)
gp_no_bmi, comp_no_bmi = load_pair_ids(PAIRS_NO_BMI_CSV)

all_gp = gp_bmi | gp_no_bmi
all_comp = comp_bmi | comp_no_bmi
all_ids = all_gp | all_comp
print(f"Total unique patients: {len(all_ids):,} "
      f"({len(all_gp):,} GP, {len(all_comp):,} comparators)")

scan_dates = {pid: surgery_dates[pid] for pid in all_ids if pid in surgery_dates}
print(f"Surgery dates found: {len(scan_dates):,}/{len(all_ids):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning medication_ingredient.csv (~75 min)...")

# {pid: [(date, drug_name, drug_category), ...]}
med_records = {}

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(MED_FILE,
                              usecols=["patient_id", "code_system", "code",
                                       "start_date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()

        rx = chunk[(chunk["code_system"] == "RxNorm") &
                   (chunk["code"].isin(ALL_CODES.keys()))].copy()
        if not rx.empty:
            rx["date"] = pd.to_datetime(rx["start_date"], errors="coerce")
            rx = rx[rx["date"].notna()]
            if not rx.empty:
                rx["drug_name"] = rx["code"].map(ALL_CODES)
                # One-time QA: confirm drug names are resolving correctly
                # (checks RxNorm ingredient-level coding vs clinical drug level)
                if chunk_num == 1:
                    print(f"  Drug name QA (first chunk): {rx['drug_name'].value_counts().to_dict()}")
                rx["category"] = rx["code"].apply(lambda c:
                    "prokinetic" if c in PROKINETIC_CODES else
                    "insulin" if c in INSULIN_CODES else
                    "metformin" if c in METFORMIN_CODES else
                    "glp1" if c in GLP1_CODES else
                    "sglt2" if c in SGLT2_CODES else "other")
                for pid, grp in rx.groupby("patient_id"):
                    existing = med_records.get(pid, [])
                    existing.extend(zip(grp["date"], grp["drug_name"], grp["category"]))
                    med_records[pid] = existing

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{len(med_records):,}/{len(all_ids):,} patients with meds found)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
print(f"  Patients with any relevant medication: {len(med_records):,}/{len(all_ids):,}")

# QA: drug frequency across all patients - confirms RxNorm codes resolved
# correctly and key drugs (semaglutide, insulin, metformin) actually appear
all_drugs = [r[1] for records in med_records.values() for r in records]
print("\nDrug frequency QA (confirms RxNorm ingredient codes matched correctly):")
print(pd.Series(all_drugs).value_counts().to_string())


def assign_med_windows(pid, group_label, surgery_date):
    group_encoded = 1 if group_label == "gastroparesis" else 0
    rows = []
    pid_records = med_records.get(pid, [])

    pid_df = pd.DataFrame(pid_records, columns=["date", "drug_name", "category"]) \
        if pid_records else pd.DataFrame(columns=["date", "drug_name", "category"])

    if not pid_df.empty:
        pid_df = pid_df.sort_values("date").reset_index(drop=True)
        pid_df["days"] = (pid_df["date"] - surgery_date).dt.days

    for tp, (d_start, d_end) in WINDOWS.items():
        # All windows now use day ranges (pre_op = -365 to -1)
        window_df = pid_df[(pid_df["days"] >= d_start) &
                            (pid_df["days"] <= d_end)] if not pid_df.empty \
            else pd.DataFrame(columns=["date", "drug_name", "category", "days"])

        for category in ["prokinetic", "insulin", "metformin", "glp1", "sglt2"]:
            cat_df = window_df[window_df["category"] == category] if not window_df.empty \
                else pd.DataFrame()
            on_med = len(cat_df) > 0
            # Preserve specific drug names for granular follow-up analysis
            drug_names = ";".join(sorted(cat_df["drug_name"].unique())) if on_med else ""
            rows.append({
                "patient_id": pid, "group": group_label,
                "group_encoded": group_encoded,
                "bariatric_date": surgery_date.date(),
                "timepoint": tp, "category": category,
                "on_medication": on_med,
                "rx_record_count": len(cat_df),  # records in TriNetX, not actual Rx count
                "drug_names": drug_names,
            })
    return rows


# --- Build outputs for both PSM versions ---
for version, gp_ids, comp_ids, label in [
    ("with_BMI", gp_bmi, comp_bmi, "63-pair"),
    ("no_BMI",   gp_no_bmi, comp_no_bmi, "94-pair"),
]:
    print(f"\nBuilding medication summary for {label} version ({version})...")
    records = []
    for pid in gp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_med_windows(pid, "gastroparesis", surg))
    for pid in comp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_med_windows(pid, "comparator", surg))

    long_df = pd.DataFrame(records)
    out_file = f"medications_long_{version}.csv"
    long_df.to_csv(out_file, index=False)

    # Sanity check: patients × timepoints × categories = expected rows
    n_patients = len(gp_ids) + len(comp_ids)
    expected_rows = n_patients * len(WINDOWS) * 5  # 5 categories
    print(f"  Wrote {out_file} ({len(long_df):,} rows, expected {expected_rows:,})")
    if len(long_df) != expected_rows:
        print(f"  ⚠ Row count mismatch - check for missing surgery dates")

    print(f"\n  Medication use summary (% on medication):")
    print(f"  {'Category':<12} {'Timepoint':<12} {'GP %':>8} {'Comp %':>8}")
    for cat in ["prokinetic", "insulin", "metformin", "glp1", "sglt2"]:
        cat_df = long_df[long_df["category"] == cat]
        for tp in WINDOWS:
            tp_df = cat_df[cat_df["timepoint"] == tp]
            gp_pct = tp_df[tp_df["group"]=="gastroparesis"]["on_medication"].mean()*100
            co_pct = tp_df[tp_df["group"]=="comparator"]["on_medication"].mean()*100
            print(f"  {cat:<12} {tp:<12} {gp_pct:>7.1f}% {co_pct:>7.1f}%")

print(f"\nTotal runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
