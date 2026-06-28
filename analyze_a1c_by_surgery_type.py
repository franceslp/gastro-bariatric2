#!/usr/bin/env python3
"""
analyze_a1c_by_surgery_type.py

Surgery-type subgroup analysis of A1c trajectories. Two analyses in one script:

ANALYSIS A — Sleeve vs Bypass A1c among matched cohort participants:
   Within each matched cohort (with_BMI, no_BMI). NOTE: this pools GP and
   comparator patients, so the sleeve-vs-bypass contrast partly reflects the
   surgery-type distribution within each study group. A separate full 376-
   patient GP sleeve-vs-bypass trajectory (a1c_trajectory_full_GP.csv) is
   available for the full-cohort analysis.

ANALYSIS B — GP vs Comparator stratified by surgery type:
   For each matched cohort, run GP-vs-comparator separately among
   SLEEVE patients and among BYPASS patients. Tests whether gastroparesis
   affects the two surgery types differently.

Surgery type: sleeve_vs_bypass (1=sleeve, 0=bypass), joined from the full
covariate matrix onto the trajectory long files.

Methods consistent with the main A1c analysis:
  - A1c LOINC already collected in trajectory files (baseline + Years 1-5)
  - Welch t-tests for between-subgroup comparisons at each year
  - Within-group change from baseline (paired)
  - Mixed model (random intercept) / OLS cluster-robust fallback where applicable
  - No imputation; available-case analysis (missingness assumed not
    informative conditional on observed data)

INPUTS:
  a1c_trajectory_with_BMI.csv, a1c_trajectory_no_BMI.csv  (long: patient_id, group, year, a1c)
  psm_full_covariate_matrix.csv        (sleeve_vs_bypass for matched patients)
  study_covariates_new.csv             (GP cohort surgery type, for full-cohort analysis A1)
  -- surgery-type analysis is limited to matched patients (those in the
     trajectory files). Full-376 trajectory would need a separate collection.

OUTPUTS:
  a1c_surgtype_sleeve_vs_bypass.csv    (Analysis A results)
  a1c_surgtype_gp_vs_comp_stratified.csv (Analysis B results)
  a1c_surgtype_summary.txt             (readable report)
  a1c_surgtype_cellcounts.csv          (n per surgery_type x group x year cell)
"""
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

# Surgery type lookup from full covariate matrix (covers both cohorts' patients)
mat = pd.read_csv("psm_full_covariate_matrix.csv", dtype={"patient_id": str})
surg_lookup = dict(zip(mat["patient_id"],
                       pd.to_numeric(mat["sleeve_vs_bypass"], errors="coerce")))

def surg_label(pid):
    v = surg_lookup.get(pid, np.nan)
    if v == 1: return "sleeve"
    if v == 0: return "bypass"
    return None

COHORTS = {"with_BMI": "a1c_trajectory_with_BMI.csv",
           "no_BMI":   "a1c_trajectory_no_BMI.csv"}

report = []
A_rows = []     # Analysis A
B_rows = []     # Analysis B
cell_rows = []  # cell counts

def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    v1, v2, n1, n2 = a.var(), b.var(), len(a), len(b)
    psd = np.sqrt(((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2))
    d = (a.mean()-b.mean())/psd if psd > 0 else np.nan
    diff = a.mean() - b.mean()
    # Welch mean-difference 95% CI (Welch-Satterthwaite dof)
    se = np.sqrt(v1/n1 + v2/n2)
    dof = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    margin = stats.t.ppf(0.975, dof) * se
    return p, d, diff, diff - margin, diff + margin

for cname, fname in COHORTS.items():
    df = pd.read_csv(fname, dtype={"patient_id": str})
    df["a1c"] = pd.to_numeric(df["a1c"], errors="coerce")
    df = df.dropna(subset=["a1c"])
    df["surg"] = df["patient_id"].map(surg_label)
    # QA: report patients dropped for missing surgery type (silent drop = bias risk)
    n_before = df["patient_id"].nunique()
    missing_surg = df[df["surg"].isna()]["patient_id"].nunique()
    if missing_surg > 0:
        report.append(f"  QA WARNING: {missing_surg} patients missing surgery type "
                     f"after merge — dropped (check covariate matrix coverage)")
    df = df.dropna(subset=["surg"])
    n_after = df["patient_id"].nunique()
    report.append(f"  Surgery type merged: {n_after}/{n_before} patients retained")

    report.append(f"\n{'='*66}\nCOHORT: {cname}\n{'='*66}")

    # Cell counts (surgery_type x group x year)
    for (st, grp, yr), g in df.groupby(["surg", "group", "year"]):
        cell_rows.append({"cohort": cname, "surgery_type": st, "group": grp,
                          "year": yr, "n": g["patient_id"].nunique()})

    # ---- ANALYSIS A (matched): Sleeve vs Bypass, pooling groups within cohort ----
    report.append(f"\n[Analysis A2 — {cname}] Sleeve vs Bypass A1c among matched participants\n  (GP+comparator pooled; contrast partly reflects surgery-type distribution by group):")
    report.append(f"  {'Year':<6}{'Sleeve mean(n)':<20}{'Bypass mean(n)':<20}{'Diff':<8}{'p':<8}{'d':<6}")
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        sl = sub[sub["surg"] == "sleeve"]["a1c"]
        by = sub[sub["surg"] == "bypass"]["a1c"]
        p, d, diff, lo, hi = welch(sl, by)
        if np.isnan(p):
            continue
        A_rows.append({"cohort": cname, "scope": "matched_pooled", "year": yr,
                       "sleeve_n": len(sl), "sleeve_mean": round(sl.mean(),3),
                       "bypass_n": len(by), "bypass_mean": round(by.mean(),3),
                       "diff_sleeve_minus_bypass": round(diff,3),
                       "diff_ci_lo": round(lo,3), "diff_ci_hi": round(hi,3),
                       "cohens_d": round(d,3), "p_value": round(p,4)})
        report.append(f"  {yr:<6}{f'{sl.mean():.2f} (n={len(sl)})':<20}"
                     f"{f'{by.mean():.2f} (n={len(by)})':<20}{diff:+.2f}   {p:.4f}  {d:+.2f}")

    # ---- ANALYSIS B: GP vs Comparator, stratified by surgery type ----
    for st in ["sleeve", "bypass"]:
        sdf = df[df["surg"] == st]
        report.append(f"\n[Analysis B — {cname} / {st.upper()}] GP vs Comparator:")
        report.append(f"  {'Year':<6}{'GP mean(n)':<18}{'Comp mean(n)':<18}{'Diff':<8}{'p':<8}{'d':<6}")
        for yr in sorted(sdf["year"].unique()):
            sub = sdf[sdf["year"] == yr]
            gp = sub[sub["group"] == "gastroparesis"]["a1c"]
            co = sub[sub["group"] == "comparator"]["a1c"]
            p, d, diff, lo, hi = welch(gp, co)
            if np.isnan(p):
                continue
            B_rows.append({"cohort": cname, "surgery_type": st, "year": yr,
                           "gp_n": len(gp), "gp_mean": round(gp.mean(),3),
                           "comp_n": len(co), "comp_mean": round(co.mean(),3),
                           "diff_gp_minus_comp": round(diff,3),
                           "diff_ci_lo": round(lo,3), "diff_ci_hi": round(hi,3),
                           "cohens_d": round(d,3), "p_value": round(p,4)})
            report.append(f"  {yr:<6}{f'{gp.mean():.2f} (n={len(gp)})':<18}"
                         f"{f'{co.mean():.2f} (n={len(co)})':<18}{diff:+.2f}   {p:.4f}  {d:+.2f}")

    # ---- Within-surgery-type change from baseline (both groups pooled) ----
    report.append(f"\n[Within-surgery-type change from baseline — {cname}]:")
    for st in ["sleeve", "bypass"]:
        sdf = df[df["surg"] == st]
        base = sdf[sdf["year"] == 0].set_index("patient_id")["a1c"]
        report.append(f"  {st}:")
        for yr in sorted(sdf["year"].unique()):
            if yr == 0: continue
            yv = sdf[sdf["year"] == yr].set_index("patient_id")["a1c"]
            common = base.index.intersection(yv.index)
            if len(common) < 2: continue
            ch = yv.loc[common] - base.loc[common]
            t, p = stats.ttest_1samp(ch, 0)
            report.append(f"    Year {yr}: Δ={ch.mean():+.3f} (n={len(common)}, p={p:.4f})")

    # ---- Mixed model: surgery_type x year, BASELINE-A1c-ADJUSTED (Point 4) ----
    # Sleeve and bypass patients may start at different A1c; adjusting for
    # baseline isolates the change attributable to surgery type rather than
    # starting severity (regression to the mean).
    if HAVE_SM:
        md = df.copy()
        md["sleeve_bin"] = (md["surg"] == "sleeve").astype(int)
        md["year_cat"] = md["year"].astype("category")
        # Merge each patient's baseline (Year 0) A1c, then model post-baseline obs
        base_map = (md[md["year"] == 0].set_index("patient_id")["a1c"].to_dict())
        md["baseline_a1c"] = md["patient_id"].map(base_map)
        # Exclude Year 0: baseline_a1c is now a PREDICTOR, so the model estimates
        # post-operative trajectory differences conditional on starting A1c.
        md_adj = md[md["year"] != 0].dropna(subset=["baseline_a1c"]).copy()
        n_drop = md["patient_id"].nunique() - md_adj["patient_id"].nunique()
        try:
            fit = None
            try:
                m1 = smf.mixedlm("a1c ~ baseline_a1c + sleeve_bin * C(year_cat)",
                                 md_adj, groups=md_adj["patient_id"])
                cand = m1.fit(method="lbfgs", disp=False)
                gv = float(cand.cov_re.iloc[0,0]) if cand.cov_re.size else 0.0
                _ = cand.conf_int()
                if gv > 1e-6:
                    fit = cand
            except Exception:
                fit = None
            if fit is None:
                fit = smf.ols("a1c ~ baseline_a1c + sleeve_bin * C(year_cat)",
                              md_adj).fit(cov_type="cluster",
                              cov_kwds={"groups": md_adj["patient_id"]})
                tag = "OLS cluster-robust (baseline-adjusted)"
            else:
                tag = "mixed random-intercept (baseline-adjusted)"
            report.append(f"\n[Sleeve-vs-Bypass mixed model — {cname}, {tag}]:")
            report.append(f"  (baseline-A1c-adjusted; {n_drop} patients without "
                         f"baseline dropped from this model)")
            report.append(f"  PRIMARY RESULT = surgery_type x year interaction (below);")
            report.append(f"  sleeve main effect (overall adj. difference) p = "
                         f"{fit.pvalues.get('sleeve_bin', np.nan):.4f}")
            ci = fit.conf_int()
            for t in [t for t in fit.pvalues.index if "sleeve_bin:" in t]:
                lo = ci.loc[t, 0] if t in ci.index else np.nan
                hi = ci.loc[t, 1] if t in ci.index else np.nan
                report.append(f"    {t}: coef={fit.params[t]:+.3f} "
                             f"[95%CI {lo:+.3f},{hi:+.3f}], p={fit.pvalues[t]:.4f}")
        except Exception as e:
            report.append(f"  surgery-type mixed model failed: {e}")

    # ---- Point 2: formal effect-modification model ----
    # Tests whether GP status modifies the A1c trajectory DIFFERENTLY by surgery
    # type. Key term = GP x sleeve x year (three-way interaction). This is the
    # formal version of Analysis B's stratified comparison.
    #
    # Because GP-vs-comparator was the MATCHED contrast (PSM balanced baseline
    # A1c), the PRIMARY three-way model is UNADJUSTED — consistent with the main
    # GP-vs-comparator trajectory analysis. A baseline-adjusted version is run as
    # a SENSITIVITY check (stratifying a matched cohort by surgery type can
    # erode within-stratum balance, so adjustment guards against that).
    if HAVE_SM:
        m3 = df.copy()
        m3["gp_bin"] = (m3["group"] == "gastroparesis").astype(int)
        m3["sleeve_bin"] = (m3["surg"] == "sleeve").astype(int)
        m3["year_cat"] = m3["year"].astype("category")
        bmap = m3[m3["year"] == 0].set_index("patient_id")["a1c"].to_dict()
        m3["baseline_a1c"] = m3["patient_id"].map(bmap)

        def fit_threeway(formula, data):
            """Fit mixed model; fall back to OLS cluster-robust if singular."""
            f = None; tag = None
            try:
                mm = smf.mixedlm(formula, data, groups=data["patient_id"])
                cand = mm.fit(method="lbfgs", disp=False)
                gv = float(cand.cov_re.iloc[0,0]) if cand.cov_re.size else 0.0
                _ = cand.conf_int()
                if gv > 1e-6:
                    f, tag = cand, "mixed random-intercept"
            except Exception:
                f = None
            if f is None:
                f = smf.ols(formula, data).fit(cov_type="cluster",
                    cov_kwds={"groups": data["patient_id"]})
                tag = "OLS cluster-robust"
            return f, tag

        def report_threeway(f3, tag3, label):
            report.append(f"\n[Effect modification — {cname}, {tag3}, {label}]:")
            report.append(f"  Three-way GP x sleeve x year terms test whether GP")
            report.append(f"  modifies the trajectory differently by surgery type:")
            tw = [t for t in f3.pvalues.index if "gp_bin:sleeve_bin:" in t]
            if tw:
                for t in tw:
                    report.append(f"    {t}: coef={f3.params[t]:+.3f}, p={f3.pvalues[t]:.4f}")
            else:
                report.append(f"    (no estimable three-way terms — cells too thin)")
            for t in [t for t in f3.pvalues.index if t == "gp_bin:sleeve_bin"]:
                report.append(f"    {t} (overall): coef={f3.params[t]:+.3f}, p={f3.pvalues[t]:.4f}")

        # PRIMARY: unadjusted (matched contrast — consistent with main analysis)
        try:
            m3_unadj = m3.copy()  # include all years; year as categorical
            f3u, tag3u = fit_threeway(
                "a1c ~ gp_bin * sleeve_bin * C(year_cat)", m3_unadj)
            report_threeway(f3u, tag3u, "PRIMARY / unadjusted")
        except Exception as e:
            report.append(f"  three-way (unadjusted) failed: {e}")

        # SENSITIVITY: baseline-adjusted (guards against within-stratum imbalance)
        try:
            m3_adj = m3[m3["year"] != 0].dropna(subset=["baseline_a1c"]).copy()
            f3a, tag3a = fit_threeway(
                "a1c ~ baseline_a1c + gp_bin * sleeve_bin * C(year_cat)", m3_adj)
            report_threeway(f3a, tag3a, "SENSITIVITY / baseline-adjusted")
        except Exception as e:
            report.append(f"  three-way (baseline-adjusted) failed: {e}")

# Point 5: baseline characteristics by surgery type (were groups comparable?)
try:
    char_rows = []
    if "age_at_surgery_approx" in mat.columns:
        for st_val, st_name in [(1, "sleeve"), (0, "bypass")]:
            g = mat[pd.to_numeric(mat["sleeve_vs_bypass"], errors="coerce") == st_val]
            row = {"surgery_type": st_name, "n": len(g)}
            for col, lab in [("age_at_surgery_approx","age"),
                             ("baseline_a1c","baseline_a1c"),
                             ("preoperative_bmi","bmi"),
                             ("diabetes_duration_log1p","dm_duration_log1p")]:
                if col in g.columns:
                    v = pd.to_numeric(g[col], errors="coerce")
                    row[f"{lab}_mean"] = round(v.mean(),2)
                    row[f"{lab}_sd"]   = round(v.std(),2)
            char_rows.append(row)
        if char_rows:
            cdf = pd.DataFrame(char_rows)
            cdf.to_csv("baseline_characteristics_by_surgery.csv", index=False)
            report.append("\n" + "="*66)
            report.append("BASELINE CHARACTERISTICS BY SURGERY TYPE (full covariate matrix)")
            report.append("="*66)
            report.append(cdf.to_string(index=False))
            report.append("  (saved to baseline_characteristics_by_surgery.csv)")
except Exception as e:
    report.append(f"\nBaseline characteristics table skipped: {e}")

pd.DataFrame(A_rows).to_csv("a1c_surgtype_sleeve_vs_bypass.csv", index=False)
pd.DataFrame(B_rows).to_csv("a1c_surgtype_gp_vs_comp_stratified.csv", index=False)
pd.DataFrame(cell_rows).to_csv("a1c_surgtype_cellcounts.csv", index=False)

text = "\n".join(report)
with open("a1c_surgtype_summary.txt", "w") as f:
    f.write(text)
print(text)
print("\n\nWrote: a1c_surgtype_sleeve_vs_bypass.csv, a1c_surgtype_gp_vs_comp_stratified.csv,")
print("       a1c_surgtype_cellcounts.csv, a1c_surgtype_summary.txt")
