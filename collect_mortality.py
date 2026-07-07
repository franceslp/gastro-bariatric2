#!/usr/bin/env python3
"""
collect_mortality.py

5-year all-cause mortality for the optimal matched with-BMI cohort (227 pairs),
following Sadda et al. JAMA Surgery 2026 (they reported 5yr all-cause mortality,
found no difference: 10.4% both groups).

Death info comes from patient.csv (month_year_death, format YYYYMM, blank if
no death recorded). This is patient-level (one row per patient) -- small file,
no chunked scan needed.

CAVEAT (state in methods/limitations): blank death field means "no death
observed," not "confirmed alive" -- TriNetX does not guarantee complete death
ascertainment, especially for Ex-US sites. This is a binary all-cause mortality
comparison (Sadda-style), not a censoring-adjusted survival analysis.

OUTPUTS:
  mortality_binary_with_BMI_optimal.csv
  mortality_summary_optimal.txt
"""
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PATIENT_FILE = f"{GCS_BASE}/patient.csv"

DATASET = "psm_matched_dataset_optimal.csv"

def load_surgery_dates():
    gp = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
    comp = pd.read_csv("comparator_pool_ready_for_PSM.csv", dtype={"patient_id": str})
    d = {}
    for _, r in gp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    for _, r in comp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    return d

def parse_month_year(series):
    """month_year_death format: YYYYMM (e.g. '202310'). Blank = no death recorded."""
    s = series.fillna("").astype(str).str.strip()
    s = s.where(s != "", None)
    # parse YYYYMM -> first day of that month, for a date to diff against surgery
    return pd.to_datetime(s, format="%Y%m", errors="coerce")

print("Loading matched cohort...")
ds = pd.read_csv(DATASET, dtype={"patient_id": str})
assert ds["patient_id"].nunique() == len(ds), "duplicate patient_ids in matched dataset"
all_ids = set(ds["patient_id"])
print(f"Matched patients: {len(all_ids)} ({(ds.group=='gastroparesis').sum()} GP, "
      f"{(ds.group=='comparator').sum()} comparator)")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}
missing_surg = [pid for pid in all_ids if pd.isna(surgery_dt[pid])]
if missing_surg:
    raise SystemExit(f"{len(missing_surg)} matched patients missing surgery date: "
                     f"{missing_surg[:5]}")

print("\nDownloading patient.csv (patient-level, ~201MB, no chunking needed)...")
proc = subprocess.run(["gsutil", "cat", PATIENT_FILE], capture_output=True)
if proc.returncode != 0:
    raise RuntimeError(f"gsutil failed (exit {proc.returncode}): "
                       f"{proc.stderr.decode(errors='replace')}")
from io import BytesIO
pat = pd.read_csv(BytesIO(proc.stdout), dtype={"patient_id": str},
                   usecols=["patient_id", "month_year_death"])
print(f"patient.csv rows: {len(pat):,}")

pat_matched = pat[pat["patient_id"].isin(all_ids)].copy()
print(f"Matched patients found in patient.csv: {len(pat_matched)}/{len(all_ids)}")
missing_from_pat = all_ids - set(pat_matched["patient_id"])
if missing_from_pat:
    # patient.csv is the master patient table -- every matched patient originated
    # from this database, so absence here indicates an export/pipeline problem,
    # not "no outcome data." Fail loudly rather than silently shrink the cohort
    # (226 GP instead of 227 would change the published denominator unnoticed).
    raise RuntimeError(
        f"{len(missing_from_pat)} matched patients NOT found in patient.csv -- "
        f"this indicates a data export/pipeline issue (patient.csv should contain "
        f"every patient), not missing outcome data. Resolve before proceeding: "
        f"{sorted(missing_from_pat)[:10]}")

assert pat["patient_id"].is_unique, "duplicate patient_id in patient.csv -- check TriNetX export"

pat_matched["death_dt"] = parse_month_year(pat_matched["month_year_death"])
death_map = dict(zip(pat_matched["patient_id"], pat_matched["death_dt"]))
group_map = dict(zip(ds["patient_id"], ds["group"]))

rows = []
for pid in all_ids:
    grp = group_map[pid]
    surg = surgery_dt[pid]
    dth = death_map.get(pid, pd.NaT)
    if pd.notna(dth):
        days_to_death = (dth - surg).days
        died_5yr = 0 <= days_to_death <= 1825
        died_ever_post = days_to_death >= 0
    else:
        days_to_death = np.nan
        died_5yr = False
        died_ever_post = False
    rows.append({"patient_id": pid, "group": grp,
                 "death_recorded": pd.notna(dth),
                 "days_to_death_post_surgery": days_to_death,
                 "died_5yr": died_5yr, "died_ever_post_surgery": died_ever_post})

mort = pd.DataFrame(rows)
mort.to_csv("mortality_binary_with_BMI_optimal.csv", index=False)

def or_ci(a_n, a_N, b_n, b_N):
    a, b, c, d = a_n, a_N - a_n, b_n, b_N - b_n
    tbl = [[a, b], [c, d]]
    try:
        if min(a, b, c, d) < 5:
            _, p = stats.fisher_exact(tbl)
        else:
            _, p, _, _ = stats.chi2_contingency(tbl)
    except Exception:
        p = np.nan
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    OR = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(np.log(OR) - 1.96*se), np.exp(np.log(OR) + 1.96*se)
    return OR, lo, hi, p

report = []
report.append("="*60)
report.append("5-YEAR ALL-CAUSE MORTALITY — GP vs Comparator (Optimal Cohort)")
report.append("Following Sadda et al. JAMA Surgery 2026 (binary, 5yr post-surgery)")
report.append("="*60)
report.append("")
report.append("CAVEAT: death ascertainment via TriNetX month_year_death field.")
report.append("Blank indicates no death recorded within TriNetX; it does not confirm")
report.append("the patient was alive throughout follow-up. Death dates are available")
report.append("only to month/year precision, so the first day of the recorded month")
report.append("was used for interval calculations (consistent with use of month-level")
report.append("administrative data; may slightly")
report.append("misclassify deaths occurring near a surgery date).")
report.append("No censoring adjustment was performed because follow-up duration could")
report.append("not be reliably determined for all patients -- this is a binary")
report.append("comparison, not a survival analysis.")
report.append("")

gp = mort[mort.group == "gastroparesis"]
co = mort[mort.group == "comparator"]

gp_d, gp_n = int(gp["died_5yr"].sum()), len(gp)
co_d, co_n = int(co["died_5yr"].sum()), len(co)
OR, lo, hi, p = or_ci(gp_d, gp_n, co_d, co_n)
report.append(f"5-year all-cause mortality:")
report.append(f"  GP: {gp_d}/{gp_n} ({100*gp_d/gp_n:.1f}%)")
report.append(f"  Comparator: {co_d}/{co_n} ({100*co_d/co_n:.1f}%)")
report.append(f"  Odds ratio (OR) = {OR:.2f} (95% CI {lo:.2f}-{hi:.2f}), p = {p:.4f}")
report.append("")

gp_de, co_de = int(gp["died_ever_post_surgery"].sum()), int(co["died_ever_post_surgery"].sum())
report.append(f"Any post-surgery death recorded (any follow-up length):")
report.append(f"  GP: {gp_de}/{gp_n} ({100*gp_de/gp_n:.1f}%)")
report.append(f"  Comparator: {co_de}/{co_n} ({100*co_de/co_n:.1f}%)")

report.append("")
report.append(f"Patients with ANY death recorded in patient.csv (matched cohort): "
              f"{int(mort['death_recorded'].sum())}/{len(mort)}")

text = "\n".join(report)
with open("mortality_summary_optimal.txt", "w") as f:
    f.write(text)
print("\n" + text)
print("\nWrote: mortality_binary_with_BMI_optimal.csv, mortality_summary_optimal.txt")
