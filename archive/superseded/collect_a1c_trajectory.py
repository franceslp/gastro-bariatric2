#!/usr/bin/env python3
"""
collect_a1c_trajectory.py

Collects post-surgery A1c trajectories (Years 1-5) for matched cohorts and
assembles full trajectories (baseline + Years 1-5) for analysis.

Follows Sadda et al. JAMA Surgery 2026 methodology:
  - Annual A1c = most recent value within each 365-day post-surgery window
  - Year 1 = days 1-365 after surgery, Year 2 = 366-730, ... Year 5 = 1461-1825
  - Baseline (Year 0) = most recent value 1-365 days BEFORE surgery
    (already collected in Phase 2; pulled from matched dataset, not re-scanned)
  - A1c LOINC: 4548-4, 17856-6, 4549-2 | plausibility 2-20%

Runs on BOTH matched cohorts in one scan:
  with-BMI  (psm_matched_pairs_new.csv,    222 pairs)
  no-BMI    (psm_matched_pairs_no_BMI.csv, 301 pairs)

OUTPUTS:
  a1c_trajectory_with_BMI.csv   (long format: patient_id, group, year, a1c)
  a1c_trajectory_no_BMI.csv
  a1c_trajectory_summary.csv    (mean/median A1c by group/year/cohort — Sadda Fig 1 style)
"""
import subprocess
import numpy as np
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE = f"{GCS_BASE}/lab_result.csv"
CHUNK = 500_000

A1C_LOINC = {"4548-4", "17856-6", "4549-2"}
A1C_MIN, A1C_MAX = 2, 20
YEARS = 5   # collect up to year 5

# Matched-pair files and their matched-dataset (for baseline + group)
COHORTS = {
    "with_BMI": {
        "pairs":   "psm_matched_pairs_new.csv",
        "dataset": "psm_matched_dataset_new.csv",
        "out":     "a1c_trajectory_with_BMI.csv",
    },
    "no_BMI": {
        "pairs":   "psm_matched_pairs_no_BMI.csv",
        "dataset": "psm_matched_dataset_no_BMI.csv",
        "out":     "a1c_trajectory_no_BMI.csv",
    },
}

# Surgery dates: GP from cohort_FINAL_analytic, comparator from
# comparator_pool_ready_for_PSM (the exact file the PSM used — guarantees the
# same surgery date the matching was built on, per reviewer Issue 1)
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

print("Loading matched cohorts...")
# Cohort-specific storage (reviewer Issue 2): a patient may be matched in BOTH
# with_BMI and no_BMI cohorts. Key group/baseline by (cohort, pid) so one cohort
# never overwrites the other.
patient_group = {}     # (cname, pid) -> group
patient_baseline = {}  # (cname, pid) -> baseline a1c (Year 0)
cohort_members = {}    # cname -> set of pids

for cname, cfg in COHORTS.items():
    ds = pd.read_csv(cfg["dataset"], dtype={"patient_id": str})

    # Issue 2: baseline_a1c must exist or trajectories silently lose Year 0
    if "baseline_a1c" not in ds.columns:
        raise ValueError(f"{cfg['dataset']} missing baseline_a1c column")
    # Issue 3: no duplicate patient_ids in the matched dataset
    assert ds["patient_id"].nunique() == len(ds), \
        f"{cname}: duplicate patient_ids in {cfg['dataset']}"

    # Issue 1: load the matched-pairs file as a cross-check on the dataset size
    pairs = pd.read_csv(cfg["pairs"], dtype=str)
    expected_patients = pairs["gp_patient_id"].nunique() + pairs["comp_patient_id"].nunique()
    print(f"  {cname}: {len(pairs)} matched pairs "
          f"(expect ~{expected_patients} patients in dataset)")

    members = set(ds["patient_id"])
    cohort_members[cname] = members
    for _, r in ds.iterrows():
        pid = r["patient_id"]
        patient_group[(cname, pid)] = r["group"]
        b = pd.to_numeric(pd.Series([r.get("baseline_a1c")]), errors="coerce").iloc[0]
        if not np.isnan(b):
            patient_baseline[(cname, pid)] = b

    # Baseline coverage sanity check: after complete-case PSM (A1c was a required
    # covariate), virtually every matched patient should have baseline A1c.
    # A low number here signals upstream breakage.
    n_baseline = sum(1 for (cn, _) in patient_baseline if cn == cname)
    pct = 100 * n_baseline / len(members) if members else 0
    print(f"  {cname}: {len(members)} matched patients | "
          f"baseline A1c present: {n_baseline}/{len(members)} ({pct:.1f}%)")
    if pct < 95:
        print(f"    WARNING: baseline A1c coverage <95% — check upstream "
              f"(A1c was a required PSM covariate, so this should be ~100%)")

# Union of all pids across cohorts (for the single lab scan)
all_ids = set()
for cname in COHORTS:
    all_ids |= cohort_members[cname]
print(f"Total unique matched patients across both cohorts: {len(all_ids)}")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}
missing_surg = [pid for pid in all_ids if pd.isna(surgery_dt[pid])]
if missing_surg:
    print(f"WARNING: {len(missing_surg)} matched patients missing surgery date")

# Post-surgery A1c store: {pid: {year: (a1c_val, date)}}
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
    # post-surgery only, within 5 years
    chunk = chunk[(chunk["days_after"] >= 1) & (chunk["days_after"] <= 365 * YEARS)]
    if chunk.empty:
        continue
    chunk["year"] = ((chunk["days_after"] - 1) // 365 + 1).astype(int)  # 1..5
    for pid, year, val, date in zip(chunk["patient_id"], chunk["year"],
                                     chunk["val"], chunk["date"]):
        cur = traj[pid].get(year)
        if cur is None or date > cur[1]:
            traj[pid][year] = (val, date)
    if rows % 50_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

# Build long-format trajectories per cohort
summary_rows = []
for cname, cfg in COHORTS.items():
    members = cohort_members[cname]
    long_rows = []
    for pid in members:
        grp = patient_group[(cname, pid)]
        # Year 0 (baseline)
        if (cname, pid) in patient_baseline:
            long_rows.append({"patient_id": pid, "group": grp, "year": 0,
                              "a1c": round(patient_baseline[(cname, pid)], 2)})
        # Years 1-5
        for yr in range(1, YEARS + 1):
            if yr in traj[pid]:
                long_rows.append({"patient_id": pid, "group": grp, "year": yr,
                                  "a1c": round(traj[pid][yr][0], 2)})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(cfg["out"], index=False)
    print(f"=== {cname} ===  wrote {cfg['out']} ({len(long_df)} patient-year rows)")

    # Wide format (reviewer Issue 3): one row per patient, columns for each year.
    # Easier for downstream responder/trajectory analysis.
    if len(long_df) > 0:
        wide = long_df.pivot(index=["patient_id", "group"],
                             columns="year", values="a1c").reset_index()
        # Rename year columns 0..5 -> baseline_a1c, year1_a1c..year5_a1c
        rename_map = {0: "baseline_a1c"}
        for y in range(1, YEARS + 1):
            rename_map[y] = f"year{y}_a1c"
        wide = wide.rename(columns=rename_map)
        # Ensure all expected columns exist even if a year had no data
        for col in ["baseline_a1c"] + [f"year{y}_a1c" for y in range(1, YEARS + 1)]:
            if col not in wide.columns:
                wide[col] = np.nan
        ordered = ["patient_id", "group", "baseline_a1c"] + \
                  [f"year{y}_a1c" for y in range(1, YEARS + 1)]
        wide = wide[ordered]
        wide_out = cfg["out"].replace(".csv", "_wide.csv")
        wide.to_csv(wide_out, index=False)
        print(f"              wrote {wide_out} ({len(wide)} patients, wide format)")

    # Summary: mean/median by group/year
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
summary.to_csv("a1c_trajectory_summary.csv", index=False)
print(f"\nwrote a1c_trajectory_summary.csv")
print("\n=== TRAJECTORY SUMMARY (mean A1c by group/year) ===")
for cname in COHORTS:
    print(f"\n{cname}:")
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
