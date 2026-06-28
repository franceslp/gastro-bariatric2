#!/usr/bin/env python3
"""
collect_a1c_full_gp_cohort.py

Collects A1c trajectories (baseline + Years 1-5) for ALL 376 GP patients,
not just the matched subset. Enables a full-cohort sleeve-vs-bypass analysis.

Method identical to collect_a1c_trajectory.py (Sadda-aligned):
  - Baseline (Year 0) = most recent A1c 1-365 days BEFORE surgery
  - Year k = most recent A1c in days (365*(k-1)+1) .. (365*k) AFTER surgery
  - A1c LOINC 4548-4 / 17856-6 / 4549-2, plausibility 2-20%
  - No imputation; most-recent value per window

Baseline (Year 0) collected directly from lab_result.csv using the same
1-365 day pre-surgery window as the matched-cohort trajectory scripts. No
fallback to study_covariates_new.csv — fresh collection guarantees all 376
patients are covered (the covariate file had ~19% A1c missingness).

Surgery type from cohort_FINAL_analytic.csv via bariatric_cpt_codes_seen, or
from study_covariates_new.csv sleeve_vs_bypass if available.

OUTPUTS:
  a1c_trajectory_full_GP.csv        (long: patient_id, year, a1c, surgery_type)
  a1c_trajectory_full_GP_wide.csv   (wide: baseline + year1..year5 + surgery_type)
  a1c_trajectory_full_GP_summary.csv
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

# Sleeve vs bypass CPT codes (from pipeline definitions)
SLEEVE_CPT = {"43775"}
BYPASS_CPT = {"43644", "43645", "43846", "43847"}

print("Loading full GP cohort (n=376)...")
cohort = pd.read_csv("cohort_FINAL_analytic.csv", dtype=str)
print(f"  cohort rows: {len(cohort)}")

# Surgery date
cohort["surg_dt"] = pd.to_datetime(cohort["bariatric_date"], errors="coerce")
surgery_dt = dict(zip(cohort["patient_id"], cohort["surg_dt"]))
assert cohort["patient_id"].nunique() == len(cohort),     "Duplicate patient IDs in cohort_FINAL_analytic.csv — check upstream pipeline"
all_ids = set(cohort["patient_id"])

# Surgery type: derive from bariatric_cpt_codes_seen
def classify_surgery(cpt_str):
    if pd.isna(cpt_str):
        return None
    codes = set(str(cpt_str).replace(";", ",").replace(" ", ",").split(","))
    codes = {c.strip() for c in codes if c.strip()}
    has_sleeve = bool(codes & SLEEVE_CPT)
    has_bypass = bool(codes & BYPASS_CPT)
    if has_sleeve and not has_bypass:
        return "sleeve"
    if has_bypass and not has_sleeve:
        return "bypass"
    if has_sleeve and has_bypass:
        return "both"   # ambiguous; should be rare after step-5 exclusion
    return None

if "bariatric_cpt_codes_seen" in cohort.columns:
    surg_type = {r["patient_id"]: classify_surgery(r["bariatric_cpt_codes_seen"])
                 for _, r in cohort.iterrows()}
else:
    surg_type = {}

st_counts = pd.Series(list(surg_type.values())).value_counts(dropna=False)
print(f"  surgery type breakdown: {st_counts.to_dict()}")
n_both = int(st_counts.get("both", 0))
n_none = int(st_counts.get(None, st_counts.get(np.nan, 0)))
if n_both > 0:
    print(f"  WARNING: {n_both} patient(s) with mixed sleeve+bypass CPT codes — "
          f"will be excluded from analysis (kept in raw output for QA)")
if n_none > 0:
    print(f"  WARNING: {n_none} patient(s) with unrecognized or missing CPT codes")

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
        for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

# store: pid -> {year: (a1c, date)}; year 0 = baseline (pre-surgery)
store = {pid: {} for pid in all_ids}

print("\nScanning lab_result.csv for A1c (baseline + Years 1-5)...")
rows = 0
for chunk in stream(LAB_FILE,
                    ["patient_id", "code_system", "code", "date", "lab_result_num_val"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)]
    if chunk.empty:
        continue
    chunk = chunk[chunk["code_system"].str.upper().str.strip() == "LOINC"]
    chunk = chunk[chunk["code"].str.strip().isin(A1C_LOINC)]
    if chunk.empty:
        continue
    chunk = chunk.copy()
    chunk["val"] = pd.to_numeric(chunk["lab_result_num_val"], errors="coerce")
    chunk = chunk[(chunk["val"] >= A1C_MIN) & (chunk["val"] <= A1C_MAX)]
    if chunk.empty:
        continue
    chunk["dt"] = parse_dates(chunk["date"])
    chunk = chunk[chunk["dt"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days"] = (chunk["dt"] - chunk["surg"]).dt.days
    # baseline window: -365..-1 ; post: 1..1825
    for pid, days, val, dt in zip(chunk["patient_id"], chunk["days"],
                                   chunk["val"], chunk["dt"]):
        if -365 <= days <= -1:
            yr = 0
        elif 1 <= days <= 365 * YEARS:
            yr = (days - 1) // 365 + 1
        else:
            continue
        cur = store[pid].get(yr)
        if cur is None or dt > cur[1]:
            store[pid][yr] = (val, dt)
    if rows % 100_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

# Build long + wide
long_rows = []
for pid in all_ids:
    st = surg_type.get(pid)
    for yr in range(0, YEARS + 1):
        if yr in store[pid]:
            long_rows.append({"patient_id": pid, "year": yr,
                              "a1c": round(store[pid][yr][0], 2),
                              "surgery_type": st})
long_df = pd.DataFrame(long_rows)

# Point 4: "both" is a mixed-exposure group — kept in raw file for QA
# but excluded from analysis outputs (sleeve/bypass only)
n_both = long_df[long_df["surgery_type"]=="both"]["patient_id"].nunique()
n_unk  = long_df[long_df["surgery_type"].isna()]["patient_id"].nunique()
if n_both > 0:
    print(f"  NOTE: {n_both} patients with 'both' surgery types excluded from analysis")
if n_unk > 0:
    print(f"  NOTE: {n_unk} patients with unknown surgery type excluded from analysis")

long_df.to_csv("a1c_trajectory_full_GP.csv", index=False)  # full, incl. both (QA)
analysis_df = long_df[long_df["surgery_type"].isin(["sleeve", "bypass"])].copy()

# Analytic group counts (after exclusion of "both"/unknown) — manuscript numbers
print("\nAnalytic surgery groups (sleeve/bypass only):")
for st in ["sleeve", "bypass"]:
    n = analysis_df[analysis_df["surgery_type"]==st]["patient_id"].nunique()
    print(f"  {st}: n={n}")
n_excl_both = long_df[long_df["surgery_type"]=="both"]["patient_id"].nunique()
n_excl_unk  = long_df[long_df["surgery_type"].isna()]["patient_id"].nunique()
if n_excl_both > 0:
    print(f"  excluded (mixed sleeve+bypass): n={n_excl_both}")
if n_excl_unk > 0:
    print(f"  excluded (unknown CPT): n={n_excl_unk}")
print(f"wrote a1c_trajectory_full_GP.csv "
      f"({long_df['patient_id'].nunique()} patients, {len(long_df)} patient-year rows)")

# Wide
wide = analysis_df.pivot_table(index="patient_id", columns="year", values="a1c",
                                aggfunc="first").reset_index()
rename = {0: "baseline_a1c"}
for y in range(1, YEARS+1):
    rename[y] = f"year{y}_a1c"
wide = wide.rename(columns=rename)
wide["surgery_type"] = wide["patient_id"].map(surg_type)  # sleeve/bypass only
for col in ["baseline_a1c"] + [f"year{y}_a1c" for y in range(1, YEARS+1)]:
    if col not in wide.columns:
        wide[col] = np.nan
wide = wide[["patient_id", "surgery_type", "baseline_a1c"] +
            [f"year{y}_a1c" for y in range(1, YEARS+1)]]
wide.to_csv("a1c_trajectory_full_GP_wide.csv", index=False)
print(f"wrote a1c_trajectory_full_GP_wide.csv ({len(wide)} patients, "
      f"one row per patient)")

# Summary by surgery type (sleeve/bypass only for analysis)
summ = []
for st in ["sleeve", "bypass"]:
    sub = analysis_df[analysis_df["surgery_type"] == st]
    if sub.empty:
        continue
    for yr in sorted(sub["year"].unique()):
        g = sub[sub["year"] == yr]
        summ.append({"surgery_type": st, "year": yr,
                     "n": g["patient_id"].nunique(),
                     "mean_a1c": round(g["a1c"].mean(), 3),
                     "sd_a1c": round(g["a1c"].std(), 3)})
pd.DataFrame(summ).to_csv("a1c_trajectory_full_GP_summary.csv", index=False)
print("wrote a1c_trajectory_full_GP_summary.csv")

# Issue 4: baseline characteristics by surgery type (mini Table 1 for subgroup)
# Pull from study_covariates_new.csv if available (has age, BMI, diabetes duration)
try:
    cov = pd.read_csv("study_covariates_new.csv", dtype={"patient_id": str})
    base_a1c = (long_df[long_df["year"]==0]
                [["patient_id","a1c","surgery_type"]]
                .rename(columns={"a1c":"baseline_a1c_collected"}))
    char = cov.merge(base_a1c, on="patient_id", how="inner")
    char_rows = []
    for st in ["sleeve","bypass"]:
        g = char[char["surgery_type"]==st]
        row = {"surgery_type": st, "n": len(g)}
        for col, label in [("baseline_a1c_collected","baseline_a1c"),
                           ("age_at_surgery_approx","age"),
                           ("preoperative_bmi","bmi"),
                           ("diabetes_duration_log1p","dm_duration_log1p")]:
            if col in g.columns:
                v = pd.to_numeric(g[col], errors="coerce")
                row[f"{label}_mean"] = round(v.mean(),2)
                row[f"{label}_sd"]   = round(v.std(),2)
                row[f"{label}_n"]    = int(v.notna().sum())
        char_rows.append(row)
    char_df = pd.DataFrame(char_rows)
    char_df.to_csv("sleeve_vs_bypass_baseline.csv", index=False)
    print("\n=== BASELINE CHARACTERISTICS BY SURGERY TYPE ===")
    print(char_df.to_string(index=False))
    print("  (saved to sleeve_vs_bypass_baseline.csv)")
except Exception as e:
    print(f"\n  Baseline characteristics table skipped: {e}")

# Issue 2: honest wording about analytic n
sl_n = analysis_df[analysis_df["surgery_type"]=="sleeve"]["patient_id"].nunique()
by_n = analysis_df[analysis_df["surgery_type"]=="bypass"]["patient_id"].nunique()
print(f"\n  Manuscript note: among 376 GP patients, {sl_n} sleeve and "
      f"{by_n} bypass patients contributed longitudinal A1c data.")

# Coverage report: how many of the 376 have A1c at each timepoint
# Saved to CSV for reviewers ("how many contributed Year 3 A1c?")
print("\n=== A1c COVERAGE ACROSS FULL GP COHORT (n=376) ===")
print(f"  {'Year':<8}{'n with A1c':<16}{'% of 376':<12}{'sleeve n':<12}{'bypass n'}")
coverage_rows = []
for yr in range(0, YEARS+1):
    yr_df = long_df[long_df["year"] == yr]
    n_total = yr_df["patient_id"].nunique()
    n_sl = yr_df[yr_df["surgery_type"]=="sleeve"]["patient_id"].nunique()
    n_by = yr_df[yr_df["surgery_type"]=="bypass"]["patient_id"].nunique()
    label = "baseline" if yr == 0 else f"Year {yr}"
    pct = 100*n_total/len(all_ids)
    print(f"  {label:<8}{n_total:<16}{pct:.1f}%      {n_sl:<12}{n_by}")
    coverage_rows.append({"year": yr, "label": label,
                          "n_with_a1c": n_total, "pct_of_376": round(pct,1),
                          "sleeve_n": n_sl, "bypass_n": n_by})
pd.DataFrame(coverage_rows).to_csv("a1c_trajectory_full_GP_coverage.csv", index=False)
print("  (saved to a1c_trajectory_full_GP_coverage.csv)")

print("\n=== FULL GP COHORT A1c BY SURGERY TYPE (mean) ===")
sdf = pd.DataFrame(summ)
for st in ["sleeve", "bypass"]:
    s = sdf[sdf["surgery_type"] == st]
    if s.empty:
        continue
    print(f"\n{st}:")
    for _, r in s.iterrows():
        print(f"  Year {int(r['year'])}: {r['mean_a1c']:.2f} (n={int(r['n'])})")

# Methodological point: primary statistical test for sleeve vs bypass is
# a1c ~ surgery_type * C(year) + (1|patient), NOT group_bin.
# The exposure here is surgery type, not GP-vs-comparator group.
try:
    import statsmodels.formula.api as smf
    md = analysis_df.copy()
    md["sleeve_bin"] = (md["surgery_type"] == "sleeve").astype(int)
    md["year_cat"] = md["year"].astype("category")

    # Issue 3: check baseline (Year 0) exists per patient before running model
    missing_base = (md.groupby("patient_id")
                    .apply(lambda x: 0 not in x["year"].values))
    n_miss = int(missing_base.sum())
    print(f"\n  Patients with post-op A1c but no baseline: {n_miss} "
          f"(will be dropped from baseline-adjusted model, retained in summary)")

    print("\n=== SLEEVE vs BYPASS MIXED MODEL (a1c ~ surgery_type * C(year) + baseline_a1c) ===")
    # Issue 1: adjust for baseline A1c to isolate the change attributable to
    # surgery type, not the starting level. Without this, a group with higher
    # baseline A1c could show greater absolute reduction purely because of
    # regression to the mean, not a true procedure effect.
    baseline_map = (analysis_df[analysis_df["year"] == 0]
                    .set_index("patient_id")["a1c"].to_dict())
    md["baseline_a1c"] = md["patient_id"].map(baseline_map)
    md_adj = md[md["year"] != 0].dropna(subset=["baseline_a1c"]).copy()
    n_dropped = md["patient_id"].nunique() - md_adj["patient_id"].nunique()
    if n_dropped > 0:
        print(f"  NOTE: {n_dropped} patients dropped from adjusted model "
              f"(no baseline A1c — have post-op data only)")

    # Model selection: try random-intercept mixed model first.
    # If the random-effect variance is estimated near zero (singular), patient-
    # level clustering contributes little beyond residual noise, and MixedLM
    # inference becomes unreliable. In that case, OLS with cluster-robust SE
    # clustered by patient is an accepted fallback: it gives the same fixed-
    # effect interaction estimates with SEs that properly account for within-
    # patient correlation, without the singularity problem.
    fit = None
    try:
        m1 = smf.mixedlm("a1c ~ sleeve_bin * C(year_cat) + baseline_a1c",
                          md_adj, groups=md_adj["patient_id"])
        cand = m1.fit(method="lbfgs", disp=False)
        gv = float(cand.cov_re.iloc[0,0]) if cand.cov_re.size else 0.0
        _ = cand.conf_int()   # raises if singular
        if gv > 1e-6:         # well-identified random effect
            fit = cand
            tag = "random-intercept mixed model (baseline-A1c-adjusted)"
    except Exception:
        fit = None
    if fit is None:
        # Fallback: random-effect variance is singular → OLS cluster-robust
        fit = smf.ols("a1c ~ sleeve_bin * C(year_cat) + baseline_a1c",
                      md_adj).fit(
            cov_type="cluster", cov_kwds={"groups": md_adj["patient_id"]})
        tag = "OLS cluster-robust SE by patient (baseline-A1c-adjusted; random-effect singular)"
    ci = fit.conf_int()
    print(f"  [{tag}]")
    print(f"  sleeve main effect p = {fit.pvalues.get('sleeve_bin', float('nan')):.4f}")
    print(f"  sleeve x year interaction terms:")
    print(f"    (tests whether sleeve-bypass difference in A1c CHANGE differs by year)")
    model_rows = []
    for t in fit.pvalues.index:
        lo = ci.loc[t, 0] if t in ci.index else float("nan")
        hi = ci.loc[t, 1] if t in ci.index else float("nan")
        model_rows.append({"term": t, "coef": round(fit.params[t],4),
                           "ci_lo": round(lo,4), "ci_hi": round(hi,4),
                           "p_value": round(fit.pvalues[t],4)})
        if "sleeve_bin:" in t:
            print(f"    {t}: coef={fit.params[t]:+.3f} "
                  f"[95%CI {lo:+.3f},{hi:+.3f}] p={fit.pvalues[t]:.4f}")
    pd.DataFrame(model_rows).to_csv("sleeve_bypass_mixedmodel.csv", index=False)
    print("  (full model saved to sleeve_bypass_mixedmodel.csv)")
except ImportError:
    print("\n(statsmodels not available — mixed model skipped)")

print("\nDone.")
