#!/usr/bin/env python3
"""
collect_a1c_trajectory_optimal.py

Same as collect_a1c_trajectory.py, but points at the OPTIMAL-matched with-BMI
cohort (227 pairs) instead of the greedy cohort. Only the COHORTS config and
output filename differ; all A1c collection logic is identical to the original,
so the trajectory is built the same way — only the matched patients change.

Run build_optimal_matched_dataset.py FIRST (creates the dataset/pairs files
this reads).

OUTPUT:
  a1c_trajectory_with_BMI_optimal.csv  (long format: patient_id, group, year, a1c)
  a1c_trajectory_with_BMI_optimal_wide.csv
"""
import subprocess
import numpy as np
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE = f"{GCS_BASE}/lab_result.csv"
CHUNK = 500_000

A1C_LOINC = {"4548-4", "17856-6", "4549-2"}
A1C_MIN, A1C_MAX = 2, 20
YEARS = 5

# OPTIMAL with-BMI cohort only
COHORTS = {
    "with_BMI_optimal": {
        "pairs":   "psm_matched_pairs_optimal.csv",
        "dataset": "psm_matched_dataset_optimal.csv",
        "out":     "a1c_trajectory_with_BMI_optimal.csv",
    },
}

def load_surgery_dates():
    gp = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
    comp = pd.read_csv("comparator_pool_ready_for_PSM.csv", dtype={"patient_id": str})
    d = {}
    for _, r in gp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    for _, r in comp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    return d

def parse_dates(series):
    series = series.fillna("").astype(str)
    r = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    m = r.isna() & (series.str.strip() != "")
    if m.any():
        r.loc[m] = pd.to_datetime(series.loc[m], format="mixed", errors="coerce")
    return r

def stream(path, usecols):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

print("Loading matched cohort (optimal with-BMI)...")
patient_group = {}
patient_baseline = {}
cohort_members = {}

for cname, cfg in COHORTS.items():
    ds = pd.read_csv(cfg["dataset"], dtype={"patient_id": str})
    if "baseline_a1c" not in ds.columns:
        raise ValueError(f"{cfg['dataset']} missing baseline_a1c column")
    assert ds["patient_id"].nunique() == len(ds), \
        f"{cname}: duplicate patient_ids in {cfg['dataset']}"

    pairs = pd.read_csv(cfg["pairs"], dtype=str)
    expected = pairs["gp_patient_id"].nunique() + pairs["comp_patient_id"].nunique()
    print(f"  {cname}: {len(pairs)} matched pairs (expect ~{expected} patients)")

    members = set(ds["patient_id"])
    cohort_members[cname] = members
    for _, r in ds.iterrows():
        pid = r["patient_id"]
        patient_group[(cname, pid)] = r["group"]
        b = pd.to_numeric(pd.Series([r.get("baseline_a1c")]), errors="coerce").iloc[0]
        if not np.isnan(b):
            patient_baseline[(cname, pid)] = b

    n_baseline = sum(1 for (cn, _) in patient_baseline if cn == cname)
    pct = 100 * n_baseline / len(members) if members else 0
    print(f"  {cname}: {len(members)} matched patients | "
          f"baseline A1c present: {n_baseline}/{len(members)} ({pct:.1f}%)")
    if pct < 95:
        print(f"    WARNING: baseline A1c coverage <95% — check upstream")

all_ids = set()
for cname in COHORTS:
    all_ids |= cohort_members[cname]
print(f"Total unique matched patients: {len(all_ids)}")

# Guard: group labels must be exactly the expected values, or the summary
# pivots (which look up "gastroparesis"/"comparator" by name) silently return
# NaN. Fail loudly instead.
bad_labels = set(patient_group.values()) - {"gastroparesis", "comparator"}
if bad_labels:
    raise SystemExit(f"Unexpected group labels: {bad_labels}. "
                     f"Expected only 'gastroparesis' / 'comparator'.")
# Guard: every matched patient must have a group label (catches silent
# dataset/membership mismatch).
assert len(all_ids) == len(patient_group), \
    (f"Cohort alignment mismatch: {len(all_ids)} matched patients but "
     f"{len(patient_group)} group labels.")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}
missing_surg = [pid for pid in all_ids if pd.isna(surgery_dt[pid])]
if missing_surg:
    # HARD FAIL (not just warn): a matched patient with no surgery date would
    # be silently dropped, misaligning the trajectory. This is the one real
    # risk when the optimal cohort includes comparators the greedy run didn't.
    raise SystemExit(
        f"{len(missing_surg)} matched patients have NO surgery date in "
        f"cohort_FINAL_analytic.csv / comparator_pool_ready_for_PSM.csv "
        f"(e.g. {missing_surg[:5]}). Resolve before proceeding — do NOT let "
        f"them silently drop from the trajectory.")
print(f"All {len(all_ids)} matched patients have surgery dates.")

traj = {pid: {} for pid in all_ids}

print("\nScanning lab_result.csv for post-surgery A1c (Years 1-5)...")
rows = 0
for chunk in stream(LAB_FILE,
                    ["patient_id", "code_system", "code", "date", "lab_result_num_val"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue
    chunk["code_system"] = chunk["code_system"].str.upper().str.strip()
    chunk = chunk[chunk["code_system"] == "LOINC"]
    chunk["code"] = chunk["code"].str.strip()
    chunk = chunk[chunk["code"].isin(A1C_LOINC)]
    if chunk.empty:
        continue
    chunk["val"] = pd.to_numeric(chunk["lab_result_num_val"], errors="coerce")
    chunk = chunk[(chunk["val"] >= A1C_MIN) & (chunk["val"] <= A1C_MAX)]
    if chunk.empty:
        continue
    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk[chunk["date"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days_after"] = (chunk["date"] - chunk["surg"]).dt.days
    chunk = chunk[(chunk["days_after"] >= 1) & (chunk["days_after"] <= 365 * YEARS)]
    if chunk.empty:
        continue
    chunk["year"] = ((chunk["days_after"] - 1) // 365 + 1).astype(int)
    for pid, year, val, date in zip(chunk["patient_id"], chunk["year"],
                                     chunk["val"], chunk["date"]):
        cur = traj[pid].get(year)
        if cur is None or date > cur[1]:
            traj[pid][year] = (val, date)

print(f"Done scanning: {rows:,} rows\n")

summary_rows = []
for cname, cfg in COHORTS.items():
    members = cohort_members[cname]
    long_rows = []
    for pid in members:
        grp = patient_group[(cname, pid)]
        if (cname, pid) in patient_baseline:
            long_rows.append({"patient_id": pid, "group": grp, "year": 0,
                              "a1c": round(patient_baseline[(cname, pid)], 2)})
        for yr in range(1, YEARS + 1):
            if yr in traj[pid]:
                long_rows.append({"patient_id": pid, "group": grp, "year": yr,
                                  "a1c": round(traj[pid][yr][0], 2)})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(cfg["out"], index=False)
    print(f"=== {cname} ===  wrote {cfg['out']} ({len(long_df)} patient-year rows)")

    if len(long_df) > 0:
        wide = long_df.pivot(index=["patient_id", "group"],
                             columns="year", values="a1c").reset_index()
        rename_map = {0: "baseline_a1c"}
        for y in range(1, YEARS + 1):
            rename_map[y] = f"year{y}_a1c"
        wide = wide.rename(columns=rename_map)
        for col in ["baseline_a1c"] + [f"year{y}_a1c" for y in range(1, YEARS + 1)]:
            if col not in wide.columns:
                wide[col] = np.nan
        ordered = ["patient_id", "group", "baseline_a1c"] + \
                  [f"year{y}_a1c" for y in range(1, YEARS + 1)]
        wide = wide[ordered]
        wide_out = cfg["out"].replace(".csv", "_wide.csv")
        wide.to_csv(wide_out, index=False)
        print(f"              wrote {wide_out} ({len(wide)} patients, wide format)")

    for (grp, yr), g in long_df.groupby(["group", "year"]):
        summary_rows.append({
            "cohort": cname, "group": grp, "year": yr,
            "n": len(g), "mean_a1c": round(g["a1c"].mean(), 3),
            "sd_a1c": round(g["a1c"].std(), 3),
            "median_a1c": round(g["a1c"].median(), 3),
        })

summary = pd.DataFrame(summary_rows)
if len(summary):
    summary = summary.sort_values(["cohort", "group", "year"])
summary.to_csv("a1c_trajectory_summary_optimal.csv", index=False)
print("\n=== TRAJECTORY SUMMARY (mean A1c by group/year) ===")
for cname in COHORTS:
    sub = summary[summary["cohort"] == cname]
    pivot = sub.pivot(index="year", columns="group", values="mean_a1c")
    ncount = sub.pivot(index="year", columns="group", values="n")
    for yr in pivot.index:
        gp_v = pivot.loc[yr].get("gastroparesis", np.nan)
        co_v = pivot.loc[yr].get("comparator", np.nan)
        gp_n = ncount.loc[yr].get("gastroparesis", 0)
        co_n = ncount.loc[yr].get("comparator", 0)
        print(f"  Year {yr}: GP {gp_v:.2f} (n={int(gp_n)})  |  "
              f"Comparator {co_v:.2f} (n={int(co_n)})")

print("\nDone.")
