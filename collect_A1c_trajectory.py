"""
collect_A1c_trajectory.py

Collects A1c trajectory for BOTH PSM versions in one lab_result.csv pass:
  - 63-pair version (with BMI): psm_matched_pairs.csv
  - 94-pair version (no BMI):   psm_matched_pairs_no_BMI.csv

Follows Sadda et al. (JAMA Surgery 2026) methodology:
  - Baseline: closest A1c before surgery (any time)
  - Year 1: days 31-365 post-surgery
  - Year 2: days 366-730
  - Year 3: days 731-1095
  - Year 4: days 1096-1460
  - Year 5: days 1461-1825
  - Within each window: most recent A1c used
  - A1c LOINC codes: 4548-4, 17856-6, 4549-2
  - Physiologic filter: 3-20%

Surgery dates pulled from master file since matched pairs CSVs
don't contain date columns.
"""

import subprocess
import time
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE = f"{GCS_BASE}/lab_result.csv"

MASTER_CSV = "master_cohort_FINAL_1118.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"
PAIRS_BMI_CSV = "psm_matched_pairs.csv"
PAIRS_NO_BMI_CSV = "psm_matched_pairs_no_BMI.csv"

A1C_LOINC_CODES = {"4548-4", "17856-6", "4549-2"}
A1C_MIN, A1C_MAX = 3.0, 20.0

WINDOWS = {
    "baseline": (None, -1),
    "year_1":   (31, 365),
    "year_2":   (366, 730),
    "year_3":   (731, 1095),
    "year_4":   (1096, 1460),
    "year_5":   (1461, 1825),
}

print(">>> SCRIPT VERSION: collect_A1c_trajectory_v1 <<<")

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
    # Get GP and comparator patient IDs
    gp_col = [c for c in df.columns if "patient_id" in c and "_gp" in c][0]
    comp_col = [c for c in df.columns if "patient_id" in c and "_comp" in c][0]
    gp_ids = set(df[gp_col].dropna())
    comp_ids = set(df[comp_col].dropna())
    return gp_ids, comp_ids

gp_bmi, comp_bmi = load_pair_ids(PAIRS_BMI_CSV)
gp_no_bmi, comp_no_bmi = load_pair_ids(PAIRS_NO_BMI_CSV)

all_ids = gp_bmi | comp_bmi | gp_no_bmi | comp_no_bmi
print(f"With-BMI version:    {len(gp_bmi):,} GP + {len(comp_bmi):,} comparators")
print(f"No-BMI version:      {len(gp_no_bmi):,} GP + {len(comp_no_bmi):,} comparators")
print(f"Total unique patients to scan: {len(all_ids):,}")

# Build surgery date lookup for all patients
scan_dates = {pid: surgery_dates[pid] for pid in all_ids if pid in surgery_dates}
n_missing = len(all_ids) - len(scan_dates)
print(f"Surgery dates found: {len(scan_dates):,}/{len(all_ids):,} (missing: {n_missing:,})")

SCRIPT_START_TIME = time.time()


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


print(f"\nScanning lab_result.csv (~178 GB, expect 90-120 min)...")

# {pid: [(date, value), ...]}
a1c_records = {}

rows_seen = 0
chunk_num = 0
for chunk in stream_gcs_csv(LAB_FILE,
                              usecols=["patient_id", "code_system", "code",
                                       "date", "value"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].notna() & chunk["patient_id"].isin(all_ids)]
    if not chunk.empty:
        chunk = chunk.copy()
        chunk["code"] = chunk["code"].str.strip()
        chunk["code_system"] = chunk["code_system"].str.strip()

        a1c = chunk[(chunk["code_system"] == "LOINC") &
                    (chunk["code"].isin(A1C_LOINC_CODES))].copy()
        if not a1c.empty:
            a1c["date"] = pd.to_datetime(a1c["date"], errors="coerce")
            a1c["value"] = pd.to_numeric(a1c["value"], errors="coerce")
            a1c = a1c[a1c["date"].notna() & a1c["value"].notna()]
            a1c = a1c[(a1c["value"] >= A1C_MIN) & (a1c["value"] <= A1C_MAX)]
            if not a1c.empty:
                for pid, grp in a1c.groupby("patient_id"):
                    existing = a1c_records.get(pid, [])
                    existing.extend(zip(grp["date"], grp["value"]))
                    a1c_records[pid] = existing

    if chunk_num % 100 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min, "
              f"{len(a1c_records):,}/{len(all_ids):,} patients with A1c)")

print(f"\n  done - scanned {rows_seen:,} rows in "
      f"{(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
print(f"  Patients with any A1c: {len(a1c_records):,}/{len(all_ids):,}")


def assign_windows(pid, group_label, surgery_date):
    group_encoded = 1 if group_label == "gastroparesis" else 0
    rows = []
    pid_records = a1c_records.get(pid, [])
    if not pid_records:
        for tp in WINDOWS:
            rows.append({"patient_id": pid, "group": group_label,
                          "group_encoded": group_encoded,
                          "bariatric_date": surgery_date.date() if pd.notna(surgery_date) else None,
                          "timepoint": tp, "group_encoded": group_encoded, "a1c_value": None,
                          "a1c_date": None, "days_from_surgery": None})
        return rows

    pid_df = pd.DataFrame(pid_records, columns=["date", "value"])
    # Sort by date for deterministic window selection - ensures "most recent"
    # is always the last row in any tie, not arbitrary insertion order
    pid_df = pid_df.sort_values("date").reset_index(drop=True)
    pid_df["days"] = (pid_df["date"] - surgery_date).dt.days

    for tp, (d_start, d_end) in WINDOWS.items():
        if tp == "baseline":
            sub = pid_df[pid_df["days"] < 0]
            if not sub.empty:
                idx = sub["days"].idxmax()
                rows.append({"patient_id": pid, "group": group_label,
                              "bariatric_date": surgery_date.date(),
                              "timepoint": tp, "group_encoded": group_encoded,
                              "a1c_value": sub.loc[idx, "value"],
                              "a1c_date": sub.loc[idx, "date"].date(),
                              "days_from_surgery": int(sub.loc[idx, "days"])})
            else:
                rows.append({"patient_id": pid, "group": group_label,
                              "bariatric_date": surgery_date.date(),
                              "timepoint": tp, "group_encoded": group_encoded, "a1c_value": None,
                              "a1c_date": None, "days_from_surgery": None})
        else:
            sub = pid_df[(pid_df["days"] >= d_start) & (pid_df["days"] <= d_end)]
            if not sub.empty:
                idx = sub["days"].idxmax()  # most recent in window per Sadda
                rows.append({"patient_id": pid, "group": group_label,
                              "bariatric_date": surgery_date.date(),
                              "timepoint": tp, "group_encoded": group_encoded,
                              "a1c_value": sub.loc[idx, "value"],
                              "a1c_date": sub.loc[idx, "date"].date(),
                              "days_from_surgery": int(sub.loc[idx, "days"])})
            else:
                rows.append({"patient_id": pid, "group": group_label,
                              "bariatric_date": surgery_date.date(),
                              "timepoint": tp, "group_encoded": group_encoded, "a1c_value": None,
                              "a1c_date": None, "days_from_surgery": None})
    return rows


# --- Build outputs for both PSM versions ---
for version, gp_ids, comp_ids, label in [
    ("with_BMI", gp_bmi, comp_bmi, "63-pair"),
    ("no_BMI",   gp_no_bmi, comp_no_bmi, "94-pair"),
]:
    print(f"\nBuilding A1c trajectory for {label} version ({version})...")
    records = []
    for pid in gp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_windows(pid, "gastroparesis", surg))
    for pid in comp_ids:
        surg = scan_dates.get(pid)
        if pd.notna(surg):
            records.extend(assign_windows(pid, "comparator", surg))

    long_df = pd.DataFrame(records)
    out_long = f"a1c_trajectory_long_{version}.csv"
    long_df.to_csv(out_long, index=False)
    print(f"  Wrote {out_long} ({len(long_df):,} rows)")

    # Summary table
    print(f"  A1c coverage by timepoint:")
    print(f"  {'Timepoint':<12} {'GP n':>6} {'GP mean':>9} {'Comp n':>8} {'Comp mean':>11}")
    for tp in WINDOWS:
        tp_df = long_df[long_df["timepoint"] == tp]
        gp_v = tp_df[(tp_df["group"]=="gastroparesis") & tp_df["a1c_value"].notna()]["a1c_value"]
        co_v = tp_df[(tp_df["group"]=="comparator") & tp_df["a1c_value"].notna()]["a1c_value"]
        print(f"  {tp:<12} {len(gp_v):>6} {gp_v.mean():>9.2f} {len(co_v):>8} {co_v.mean():>11.2f}")

print(f"\nTotal runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
