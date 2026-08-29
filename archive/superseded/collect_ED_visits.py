"""
collect_ED_visits.py

Collects post-operative ED visit counts for BOTH PSM versions in one
encounter.csv pass:
  - 63-pair version (with BMI): psm_matched_pairs.csv
  - 94-pair version (no BMI):   psm_matched_pairs_no_BMI.csv

Follows same 5-year window structure as A1c trajectory (Sadda et al.):
  - Year 1: days 31-365 post-surgery
  - Year 2: days 366-730
  - Year 3: days 731-1095
  - Year 4: days 1096-1460
  - Year 5: days 1461-1825

ED visits defined as encounter type = 'EMER' (confirmed from dataset).
Also captures 'IMP' (inpatient) as a secondary outcome.

Output: one row per patient per timepoint per outcome type (long format),
with counts (not just presence/absence) per window.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
ENCOUNTER_FILE = f"{GCS_BASE}/encounter.csv"

MASTER_CSV = "master_cohort_FINAL_1118.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
PAIRS_BMI_CSV = "psm_matched_pairs.csv"
PAIRS_NO_BMI_CSV = "psm_matched_pairs_no_BMI.csv"

ED_TYPE = "EMER"
IP_TYPE = "IMP"

WINDOWS = {
    "year_1":  (31, 365),
    "year_2":  (366, 730),
    "year_3":  (731, 1095),
    "year_4":  (1096, 1460),
    "year_5":  (1461, 1825),
}

print(">>> SCRIPT VERSION: collect_ED_visits_v1 <<<")

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

all_ids = gp_bmi | comp_bmi | gp_no_bmi | comp_no_bmi
print(f"With-BMI version:    {len(gp_bmi):,} GP + {len(comp_bmi):,} comparators")
print(f"No-BMI version:      {len(gp_no_bmi):,} GP + {len(comp_no_bmi):,} comparators")
print(f"Total unique patients to scan: {len(all_ids):,}")

scan_dates = {pid: surgery_dates[pid] for pid in all_ids if pid in surgery_dates}
print(f"Surgery dates found: {len(scan_dates):,}/{len(all_ids):,}")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning encounter.csv (~32 GB, expect 15-20 min)...")

# {pid: [(date, type), ...]} - only ED and inpatient
encounter_records = {}

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(ENCOUNTER_FILE,
                              usecols=["patient_id", "start_date", "type"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["type"] = chunk["type"].str.strip()

        # Keep only ED and inpatient encounters
        relevant = chunk[chunk["type"].isin([ED_TYPE, IP_TYPE])].copy()
        if not relevant.empty:
            if chunk_num == 1:
                print(f"  Encounter type sample: {chunk['type'].value_counts().to_dict()}")
            relevant["date"] = pd.to_datetime(relevant["start_date"], errors="coerce")
            relevant = relevant[relevant["date"].notna()]
            if not relevant.empty:
                for pid, grp in relevant.groupby("patient_id"):
                    existing = encounter_records.get(pid, [])
                    existing.extend(zip(grp["date"], grp["type"]))
                    encounter_records[pid] = existing

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{len(encounter_records):,}/{len(all_ids):,} patients with ED/IP visits)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
print(f"  Patients with any ED or inpatient visit: {len(encounter_records):,}/{len(all_ids):,}")


def assign_encounter_windows(pid, group_label, surgery_date):
    group_encoded = 1 if group_label == "gastroparesis" else 0
    rows = []
    pid_records = encounter_records.get(pid, [])

    if not pid_records:
        for tp in WINDOWS:
            for enc_type in [ED_TYPE, IP_TYPE]:
                rows.append({"patient_id": pid, "group": group_label,
                              "group_encoded": group_encoded,
                              "bariatric_date": surgery_date.date(),
                              "timepoint": tp, "encounter_type": enc_type,
                              "count": 0, "had_visit": False})
        return rows

    pid_df = pd.DataFrame(pid_records, columns=["date", "type"])
    pid_df = pid_df.sort_values("date").reset_index(drop=True)
    pid_df["days"] = (pid_df["date"] - surgery_date).dt.days

    for tp, (d_start, d_end) in WINDOWS.items():
        window_df = pid_df[(pid_df["days"] >= d_start) & (pid_df["days"] <= d_end)]
        for enc_type in [ED_TYPE, IP_TYPE]:
            type_df = window_df[window_df["type"] == enc_type]
            count = len(type_df)
            rows.append({"patient_id": pid, "group": group_label,
                          "group_encoded": group_encoded,
                          "bariatric_date": surgery_date.date(),
                          "timepoint": tp, "encounter_type": enc_type,
                          "count": count, "had_visit": count > 0})
    return rows


# --- Build outputs for both PSM versions ---
for version, gp_ids, comp_ids, label in [
    ("with_BMI", gp_bmi, comp_bmi, "63-pair"),
    ("no_BMI",   gp_no_bmi, comp_no_bmi, "94-pair"),
]:
    print(f"\nBuilding ED/IP visit counts for {label} version ({version})...")
    records = []
    for pid in gp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_encounter_windows(pid, "gastroparesis", surg))
    for pid in comp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_encounter_windows(pid, "comparator", surg))

    long_df = pd.DataFrame(records)
    out_file = f"ed_visits_long_{version}.csv"
    long_df.to_csv(out_file, index=False)
    print(f"  Wrote {out_file} ({len(long_df):,} rows)")

    # Summary by timepoint
    for enc_type, label_str in [(ED_TYPE, "ED (EMER)"), (IP_TYPE, "Inpatient (IMP)")]:
        print(f"\n  {label_str} visits:")
        print(f"  {'Timepoint':<12} {'GP any':>8} {'GP mean':>9} {'Comp any':>10} {'Comp mean':>11}")
        type_df = long_df[long_df["encounter_type"] == enc_type]
        for tp in WINDOWS:
            tp_df = type_df[type_df["timepoint"] == tp]
            gp_df = tp_df[tp_df["group"] == "gastroparesis"]
            co_df = tp_df[tp_df["group"] == "comparator"]
            print(f"  {tp:<12} "
                  f"{gp_df['had_visit'].sum():>8} {gp_df['count'].mean():>9.2f} "
                  f"{co_df['had_visit'].sum():>10} {co_df['count'].mean():>11.2f}")

print(f"\nTotal runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
