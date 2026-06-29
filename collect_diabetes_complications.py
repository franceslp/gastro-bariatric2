#!/usr/bin/env python3
"""
collect_diabetes_complications.py

Collects diabetes-related complications for matched GP-bariatric vs comparator
cohorts, following Sadda et al. JAMA Surgery 2026 methodology.

COMPLICATIONS (from Sadda eTable 1):
  Renal:       CKD (N18), ESRD (N18.6, Z99.2), Kidney transplant (Z94.0, T86.1x)
  Circulatory: Heart failure (I50), MI (I21), Stroke (I63)
  Ophthalmic:  Retinopathy (E10.31-35, E11.31-35), Glaucoma (H40-42)
  Glycemic:    DKA/HHS (E10.1, E11.0, E11.1)
  NOTE: Hypoglycemia (glucose <54) requires lab_result.csv — handled separately.

OUTCOMES reported at:
  Short-term: 1 year post-surgery (days 1-365)
  Long-term:  5 years post-surgery (days 1-1825)
  (Sadda: follow-up begins 1 month post-index; we use day 1 to capture
   all post-op complications, consistent with Sadda's complication reporting)

PATIENTS INCLUDED: only those WITHOUT the complication at baseline
  (1-365 days before surgery), per Sadda methodology.

PRIMARY statistic: binary (new-onset) OR with 95% CI + chi-square
  Reported at 1yr and 5yr, GP vs comparator.

Runs on BOTH matched cohorts in one diagnosis.csv scan.

OUTPUTS:
  dm_complications_binary_with_BMI.csv   (per patient: new-onset flags)
  dm_complications_binary_no_BMI.csv
  dm_complications_summary.txt           (OR table, Sadda Table 2 style)
"""
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAG_FILE = f"{GCS_BASE}/diagnosis.csv"
CHUNK = 500_000

# Sadda eTable 1 complication codes — ICD-10 prefix matching
COMPLICATIONS = {
    # Renal
    "ckd":               {"prefixes": ["N18"],     "exclude": []},
    "esrd":              {"prefixes": ["N18.6","Z99.2"], "exclude": []},
    "kidney_transplant": {"prefixes": ["Z94.0","T86.1"], "exclude": []},
    # Circulatory
    "heart_failure":     {"prefixes": ["I50"],     "exclude": []},
    "mi":                {"prefixes": ["I21"],      "exclude": []},
    "stroke":            {"prefixes": ["I63"],      "exclude": []},
    # Ophthalmic
    # Sadda eTable 1: diabetic retinopathy specifically (nonproliferative
    # and proliferative); E10.3/E11.3 alone is too broad (catches unspecified
    # eye disease). Using E10.31-35, E11.31-35 per Sadda supplement.
    "retinopathy":       {"prefixes": [
        "E10.31","E10.32","E10.33","E10.34","E10.35",
        "E11.31","E11.32","E11.33","E11.34","E11.35",
    ], "exclude": []},
    "glaucoma":          {"prefixes": ["H40","H41","H42"], "exclude": []},
    # Glycemic
    # Renamed hyperglycemic_crisis (covers DKA: E10.1/E11.1 and
    # HHS: E11.0) — avoids misleading DKA/HHS label in publication.
    "hyperglycemic_crisis": {"prefixes": ["E10.1","E11.0","E11.1"], "exclude": []},
    # Composite: uses same narrow retinopathy codes as individual outcome
    # to ensure consistent definition across individual and composite analyses
    "any_complication":  {"prefixes": [
        "N18","Z99.2","Z94.0","T86.1",
        "I50","I21","I63",
        "E10.31","E10.32","E10.33","E10.34","E10.35",
        "E11.31","E11.32","E11.33","E11.34","E11.35",
        "H40","H41","H42",
        "E10.1","E11.0","E11.1"
    ], "exclude": []},
}

COHORTS = {
    "with_BMI": {"dataset": "psm_matched_dataset_new.csv"},
    "no_BMI":   {"dataset": "psm_matched_dataset_no_BMI.csv"},
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
    s = series.fillna("").astype(str)
    r = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    m = r.isna() & (s.str.strip() != "")
    if m.any():
        r.loc[m] = pd.to_datetime(s.loc[m], format="mixed", errors="coerce")
    return r

def stream(path, usecols):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(proc.stdout, usecols=usecols,
                                  dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

def code_matches(code, prefixes):
    """True if ICD-10 code starts with any of the prefixes."""
    code = str(code).strip().upper()
    return any(code.startswith(p.upper()) for p in prefixes)

print("Loading matched cohorts...")
all_ids = set()
cohort_members = {}
for cname, cfg in COHORTS.items():
    ds = pd.read_csv(cfg["dataset"], dtype={"patient_id": str})
    assert ds["patient_id"].nunique() == len(ds), \
        f"{cname}: duplicate patient_ids"
    members = {r["patient_id"]: r["group"] for _, r in ds.iterrows()}
    cohort_members[cname] = members
    all_ids |= set(members.keys())
    print(f"  {cname}: {len(members)} matched patients")
print(f"Total unique patients: {len(all_ids)}")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}

# Load follow-up days from ED/hospitalization output.
# IMPORTANT: followup_days_post in that file was derived from the LAST
# ENCOUNTER OF ANY TYPE (not just ED/IP) during the encounter.csv scan —
# specifically the max day-offset of any encounter post-surgery. This is
# a valid total-observation proxy and is NOT circular with the complication
# outcome (which comes from diagnosis.csv). Document this in methods:
# "Post-operative follow-up was estimated as the interval from surgery to
#  the last recorded clinical encounter of any type."
followup_days = {}
for fu_file in ["ed_hosp_binary_with_BMI.csv", "ed_hosp_binary_no_BMI.csv"]:
    try:
        fu_df = pd.read_csv(fu_file, dtype={"patient_id": str})
        if "followup_days_post" in fu_df.columns:
            for _, r in fu_df.iterrows():
                pid = r["patient_id"]
                val = pd.to_numeric(r["followup_days_post"], errors="coerce")
                if pid not in followup_days or (not pd.isna(val) and
                   (pd.isna(followup_days.get(pid, float("nan"))) or
                    val > followup_days[pid])):
                    followup_days[pid] = val
    except FileNotFoundError:
        pass
n_with_fu = sum(1 for v in followup_days.values() if not pd.isna(v))
n_full5yr = sum(1 for v in followup_days.values()
                if not pd.isna(v) and v >= 1825)
print(f"Follow-up loaded: {n_with_fu} patients, "
      f"{n_full5yr} with >=5yr follow-up")

# Store: pid -> {comp_name: [days_after_surgery, ...]}
# days can be negative (baseline) or positive (post-op)
store = {pid: {comp: [] for comp in COMPLICATIONS} for pid in all_ids}

date_min = None
date_max = None
print(f"\nScanning diagnosis.csv for {len(COMPLICATIONS)} complication categories...")
rows = 0
for chunk in stream(DIAG_FILE, ["patient_id", "code", "date"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue
    chunk["dt"] = parse_dates(chunk["date"])
    chunk = chunk[chunk["dt"].notna()]
    # Track date range for QA (TriNetX date parsing issues caught this before)
    if len(chunk) > 0:
        cmin = chunk["dt"].min(); cmax = chunk["dt"].max()
        if date_min is None or cmin < date_min: date_min = cmin
        if date_max is None or cmax > date_max: date_max = cmax
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days"] = (chunk["dt"] - chunk["surg"]).dt.days
    # keep only -365 (baseline) to +1825 (5yr post)
    chunk = chunk[(chunk["days"] >= -365) & (chunk["days"] <= 1825)]
    if chunk.empty:
        continue
    chunk["code"] = chunk["code"].str.strip().str.upper()
    for pid, code, days in zip(chunk["patient_id"],
                                chunk["code"], chunk["days"]):
        for comp, cfg in COMPLICATIONS.items():
            if code_matches(code, cfg["prefixes"]):
                store[pid][comp].append(int(days))
    if rows % 100_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

def or_ci(gp_n, gp_N, co_n, co_N):
    a,b,c,d = gp_n, gp_N-gp_n, co_n, co_N-co_n
    tbl = np.array([[a,b],[c,d]])
    # Issue 5: use Fisher exact for rare outcomes (any cell < 5),
    # chi-square otherwise (more defensible for sparse data).
    try:
        if min(a,b,c,d) < 5:
            _, p = stats.fisher_exact(tbl)
        else:
            _, p, _, _ = stats.chi2_contingency(tbl)
    except Exception:
        p = np.nan
    # Haldane correction for zero cells before OR calculation
    if min(a,b,c,d) == 0:
        a,b,c,d = a+.5,b+.5,c+.5,d+.5
    OR = (a*d)/(b*c)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    lo,hi = np.exp(np.log(OR)-1.96*se), np.exp(np.log(OR)+1.96*se)
    return OR, lo, hi, p

report = []
report.append("="*72)
report.append("DIABETES COMPLICATIONS — GP vs Comparator")
report.append("Following Sadda et al. JAMA Surgery 2026 (eTable 1 codes)")
report.append("New-onset only (patients without complication at baseline excluded)")
report.append("Short-term = 1yr (days 1-365); Long-term = 5yr (days 1-1825)")
report.append("="*72)

for cname, cfg in COHORTS.items():
    members = cohort_members[cname]
    report.append(f"\n{'='*60}\nCOHORT: {cname}\n{'='*60}")

    binary_rows = []
    for pid, grp in members.items():
        row = {"patient_id": pid, "group": grp}
        for comp in COMPLICATIONS:
            days_list = store[pid][comp]
            # baseline: any occurrence in -365 to -1
            at_baseline = any(-365 <= d <= -1 for d in days_list)
            # new-onset 1yr: first occurrence in days 1-365 (no baseline)
            new_1yr = (not at_baseline and
                       any(1 <= d <= 365 for d in days_list))
            # new-onset 5yr: first occurrence in days 1-1825 (no baseline)
            new_5yr = (not at_baseline and
                       any(1 <= d <= 1825 for d in days_list))
            # strict 5yr: only count if patient had >=1825 days follow-up
            fu = followup_days.get(pid, float("nan"))
            has_full5yr = (not pd.isna(fu) and fu >= 1825)
            row[f"{comp}_baseline"]       = at_baseline
            row[f"{comp}_new_1yr"]        = new_1yr
            row[f"{comp}_new_5yr"]        = new_5yr       # available-case
            row[f"{comp}_new_5yr_strict"] = (new_5yr and has_full5yr)  # strict
        binary_rows.append(row)

    bdf = pd.DataFrame(binary_rows)
    bdf.to_csv(f"dm_complications_binary_{cname}.csv", index=False)

    # ---- Report: Sadda Table 2 style ----
    # Table: 1yr | 5yr available-case (AC) | 5yr strict (S, >=1825d follow-up)
    # 5yr strict is the methodologically correct primary 5yr endpoint.
    # 5yr AC is reported for transparency.
    report.append(f"\n{'Complication':<22} "
                 f"{'GP 1yr':>9} {'Co 1yr':>9} {'OR(CI)':>16} {'p':>6}  "
                 f"{'GP 5AC':>8} {'Co 5AC':>8} {'OR(CI)':>16} {'p':>6}  "
                 f"{'GP 5S':>7} {'Co 5S':>7} {'OR(CI)':>16} {'p':>6}")
    report.append("-"*150)

    def cell(gn, gN, cn, cN):
        OR, lo, hi, p = or_ci(gn, gN, cn, cN)
        return (f"{gn}/{gN}({100*gn/gN if gN else 0:.0f}%)",
                f"{cn}/{cN}({100*cn/cN if cN else 0:.0f}%)",
                f"{OR:.2f}({lo:.2f}-{hi:.2f})", f"{p:.4f}")

    for comp in COMPLICATIONS:
        gp_sub = bdf[(bdf["group"]=="gastroparesis") & (~bdf[f"{comp}_baseline"])]
        co_sub = bdf[(bdf["group"]=="comparator")    & (~bdf[f"{comp}_baseline"])]

        # 1yr
        r1 = cell(gp_sub[f"{comp}_new_1yr"].sum(), len(gp_sub),
                  co_sub[f"{comp}_new_1yr"].sum(), len(co_sub))
        # 5yr available-case (all patients regardless of follow-up)
        r5ac = cell(gp_sub[f"{comp}_new_5yr"].sum(), len(gp_sub),
                    co_sub[f"{comp}_new_5yr"].sum(), len(co_sub))
        # 5yr strict: only patients with >=1825 days follow-up
        # This is the methodologically correct 5yr endpoint per Issue 2
        gp_s = gp_sub[~gp_sub[f"{comp}_new_5yr_strict"].isna()]
        co_s = co_sub[~co_sub[f"{comp}_new_5yr_strict"].isna()]
        r5s = cell(gp_s[f"{comp}_new_5yr_strict"].sum(), len(gp_s),
                   co_s[f"{comp}_new_5yr_strict"].sum(), len(co_s))

        report.append(
            f"{comp:<22} "
            f"{r1[0]:>9} {r1[1]:>9} {r1[2]:>16} {r1[3]:>6}  "
            f"{r5ac[0]:>8} {r5ac[1]:>8} {r5ac[2]:>16} {r5ac[3]:>6}  "
            f"{r5s[0]:>7} {r5s[1]:>7} {r5s[2]:>16} {r5s[3]:>6}")

    print(f"{cname}: wrote dm_complications_binary_{cname}.csv")

text = "\n".join(report)
with open("dm_complications_summary.txt","w") as f:
    f.write(text)
print(text)
print("\nWrote: dm_complications_summary.txt")
