#!/usr/bin/env python3
"""
analyze_a1c_trajectory_optimal.py

IDENTICAL statistics to analyze_a1c_trajectory.py — only the COHORTS input is
changed to the OPTIMAL-matched with-BMI trajectory
(a1c_trajectory_with_BMI_optimal.csv, 227 pairs). All models (per-year Welch
t-tests with FDR, mixed-effects group*year interaction, baseline-adjusted
sensitivity, responder analysis) are unchanged, so results are directly
comparable to the greedy run.

Run AFTER collect_a1c_trajectory_optimal.py has produced
a1c_trajectory_with_BMI_optimal.csv.

OUTPUTS (suffixed _optimal so they don't overwrite the greedy results):
  a1c_stats_timepoint_optimal.csv
  a1c_stats_mixedmodel_optimal.csv
  a1c_stats_coverage_optimal.csv
  a1c_stats_summary_optimal.txt
"""
import numpy as np
import pandas as pd
from scipy import stats

# OPTIMAL with-BMI cohort only
COHORTS = {
    "with_BMI_optimal": "a1c_trajectory_with_BMI_optimal.csv",
}

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

timepoint_rows = []
mixed_rows = []
report_lines = []

report_lines.append("="*64)
report_lines.append("A1c TRAJECTORY ANALYSIS (OPTIMAL cohort) — GP-bariatric vs matched comparator")
report_lines.append("="*64)
report_lines.append("PRIMARY endpoint: group x time interaction (mixed-effects model)")
report_lines.append("SECONDARY: annual between-group A1c differences (Years 1-5);")
report_lines.append("           responder analysis (>=1.1% A1c decrease from baseline).")
report_lines.append("")
report_lines.append("Matching: optimal 1:1, logit distance, 0.2 SD logit caliper,")
report_lines.append("          unpenalized logistic PS (227 pairs, with-BMI cohort).")
report_lines.append("")
report_lines.append("Methods notes:")
report_lines.append("- Mixed models use all available observations under a")
report_lines.append("  missing-at-random (MAR) assumption; no imputation.")
report_lines.append("- Random intercept by patient_id accounts for within-patient")
report_lines.append("  repeated measures. Matched-pair dependence was addressed via")
report_lines.append("  propensity-score balancing; pair-level random effects were not")
report_lines.append("  modeled (acceptable given sample size; stated as a limitation).")
report_lines.append("- Annual t-tests are Welch (unequal variance); post-baseline")
report_lines.append("  comparisons FDR-corrected (Benjamini-Hochberg).")

def ci95(mean, sd, n):
    if n < 2 or np.isnan(sd):
        return (np.nan, np.nan)
    se = sd / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return (mean - h, mean + h)

for cname, fname in COHORTS.items():
    df = pd.read_csv(fname, dtype={"patient_id": str})
    df["a1c"] = pd.to_numeric(df["a1c"], errors="coerce")
    df = df.dropna(subset=["a1c"])

    report_lines.append(f"\n{'='*64}\nCOHORT: {cname}\n{'='*64}")
    obs_counts = df.groupby("patient_id").size()
    report_lines.append(f"Patients with >=2 A1c observations (contribute to mixed model): "
                       f"{int((obs_counts >= 2).sum())}/{df['patient_id'].nunique()}")

    # --- 1. Per-timepoint t-tests ---
    report_lines.append("\nPer-timepoint comparison (independent-samples t-test):")
    report_lines.append(f"{'Year':<6}{'GP mean(SD) n':<22}{'Comp mean(SD) n':<22}"
                        f"{'Diff':<8}{'p-value':<10}")
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        gp = sub[sub["group"] == "gastroparesis"]["a1c"]
        co = sub[sub["group"] == "comparator"]["a1c"]
        if len(gp) < 2 or len(co) < 2:
            continue
        t, p = stats.ttest_ind(gp, co, equal_var=False)
        diff = gp.mean() - co.mean()
        gp_lo, gp_hi = ci95(gp.mean(), gp.std(), len(gp))
        co_lo, co_hi = ci95(co.mean(), co.std(), len(co))
        v_gp, v_co = gp.var(), co.var()
        n_gp, n_co = len(gp), len(co)
        se_diff = np.sqrt(v_gp/n_gp + v_co/n_co)
        dof = ((v_gp/n_gp + v_co/n_co)**2 /
               ((v_gp/n_gp)**2/(n_gp-1) + (v_co/n_co)**2/(n_co-1)))
        margin = stats.t.ppf(0.975, dof) * se_diff
        diff_lo, diff_hi = diff - margin, diff + margin
        pooled_sd = np.sqrt(((n_gp-1)*v_gp + (n_co-1)*v_co) / (n_gp + n_co - 2))
        cohens_d = diff / pooled_sd if pooled_sd > 0 else np.nan
        timepoint_rows.append({
            "cohort": cname, "year": yr,
            "gp_n": len(gp), "gp_mean": round(gp.mean(),3), "gp_sd": round(gp.std(),3),
            "gp_ci_lo": round(gp_lo,3), "gp_ci_hi": round(gp_hi,3),
            "comp_n": len(co), "comp_mean": round(co.mean(),3), "comp_sd": round(co.std(),3),
            "comp_ci_lo": round(co_lo,3), "comp_ci_hi": round(co_hi,3),
            "diff_gp_minus_comp": round(diff,3),
            "diff_ci_lo": round(diff_lo,3), "diff_ci_hi": round(diff_hi,3),
            "cohens_d": round(cohens_d,3),
            "welch_dof": round(dof,1),
            "t_stat": round(t,3), "p_value": round(p,4),
        })
        gp_str = f"{gp.mean():.2f}({gp.std():.2f}) n={len(gp)}"
        co_str = f"{co.mean():.2f}({co.std():.2f}) n={len(co)}"
        sig = "  *" if p < 0.05 else ""
        report_lines.append(f"{yr:<6}{gp_str:<22}{co_str:<22}{diff:+.2f}   {p:.4f}{sig}")

    try:
        from statsmodels.stats.multitest import multipletests
        cohort_rows = [r for r in timepoint_rows if r["cohort"] == cname and r["year"] != 0]
        if cohort_rows:
            pvals = [r["p_value"] for r in cohort_rows]
            p_adj = multipletests(pvals, method="fdr_bh")[1]
            for r, pa in zip(cohort_rows, p_adj):
                r["p_adjusted_fdr"] = round(pa, 4)
                r["significant_05_adj"] = pa < 0.05
            report_lines.append("\n  Post-baseline raw vs FDR-adjusted p:")
            for r in cohort_rows:
                star = " *" if r.get("significant_05_adj") else ""
                report_lines.append(
                    f"    Year {r['year']}: diff={r['diff_gp_minus_comp']:+.2f} "
                    f"(d={r['cohens_d']:+.2f}), raw p={r['p_value']:.4f}, "
                    f"FDR p={r['p_adjusted_fdr']:.4f}{star}")
    except ImportError:
        pass

    # --- 2. Within-group change from baseline ---
    report_lines.append("\nWithin-group paired analysis (patients with both baseline and follow-up A1c):")
    for grp in ["gastroparesis", "comparator"]:
        g = df[df["group"] == grp]
        base = g[g["year"] == 0].set_index("patient_id")["a1c"]
        report_lines.append(f"  {grp}:")
        for yr in sorted(g["year"].unique()):
            if yr == 0:
                continue
            yv = g[g["year"] == yr].set_index("patient_id")["a1c"]
            common = base.index.intersection(yv.index)
            if len(common) < 2:
                continue
            change = (yv.loc[common] - base.loc[common])
            t, p = stats.ttest_1samp(change, 0)
            report_lines.append(f"    Year {yr}: Δ={change.mean():+.3f} "
                               f"(n={len(common)} paired, p={p:.4f})")

    # --- 2b. Responder analysis ---
    report_lines.append("\nResponder analysis (>=1.1% A1c decrease from baseline):")
    RESPONDER_THRESHOLD = 1.1
    for yr in sorted([y for y in df["year"].unique() if y != 0]):
        prop = {}
        counts = {}
        for grp in ["gastroparesis", "comparator"]:
            g = df[df["group"] == grp]
            base = g[g["year"] == 0].set_index("patient_id")["a1c"]
            yv = g[g["year"] == yr].set_index("patient_id")["a1c"]
            common = base.index.intersection(yv.index)
            if len(common) < 2:
                prop[grp] = np.nan; counts[grp] = (0, 0); continue
            decrease = base.loc[common] - yv.loc[common]
            n_resp = int((decrease >= RESPONDER_THRESHOLD).sum())
            prop[grp] = n_resp / len(common)
            counts[grp] = (n_resp, len(common))
        gp_c, co_c = counts.get("gastroparesis"), counts.get("comparator")
        if gp_c and co_c and gp_c[1] >= 2 and co_c[1] >= 2:
            table = np.array([[gp_c[0], gp_c[1]-gp_c[0]],
                              [co_c[0], co_c[1]-co_c[0]]])
            try:
                chi2, pchi, _, _ = stats.chi2_contingency(table)
            except Exception:
                pchi = np.nan
            report_lines.append(
                f"  Year {yr}: GP {gp_c[0]}/{gp_c[1]} ({100*prop['gastroparesis']:.1f}%) "
                f"vs Comp {co_c[0]}/{co_c[1]} ({100*prop['comparator']:.1f}%)  "
                f"chi2 p={pchi:.4f}")

    # --- 3. Mixed-effects model ---
    if HAVE_SM:
        md = df.copy()
        md["group_bin"] = (md["group"] == "gastroparesis").astype(int)
        md["year_cat"] = md["year"].astype("category")
        try:
            import statsmodels.formula.api as smf_ols
            fit = None
            use_mixed = False
            try:
                model = smf.mixedlm("a1c ~ group_bin * C(year_cat)", md,
                                    groups=md["patient_id"])
                cand = model.fit(method="lbfgs", disp=False)
                grp_var = float(cand.cov_re.iloc[0, 0]) if cand.cov_re.size else 0.0
                _ = cand.conf_int()
                if grp_var > 1e-6:
                    fit = cand
                    use_mixed = True
            except Exception:
                fit = None
            if fit is None:
                fit = smf_ols.ols("a1c ~ group_bin * C(year_cat)", md).fit(
                    cov_type="cluster", cov_kwds={"groups": md["patient_id"]})
            method_label = ("random-intercept mixed model" if use_mixed
                            else "OLS with cluster-robust SE by patient "
                                 "(random-intercept variance was singular)")
            report_lines.append(f"\n  [{method_label}]")

            ci = fit.conf_int()
            for param in fit.params.index:
                lo = ci.loc[param, 0] if param in ci.index else np.nan
                hi = ci.loc[param, 1] if param in ci.index else np.nan
                mixed_rows.append({
                    "cohort": cname, "model": "primary_allparams", "term": param,
                    "coef": round(fit.params[param], 4),
                    "se": round(fit.bse[param], 4),
                    "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                    "p_value": round(fit.pvalues[param], 4),
                })
            inter_terms = [t for t in fit.pvalues.index if "group_bin:" in t]
            report_lines.append(f"\nPRIMARY model (a1c ~ group * C(year)):")
            report_lines.append(f"  group main effect (baseline group difference) p = "
                               f"{fit.pvalues.get('group_bin', np.nan):.4f}")
            report_lines.append(f"  group x year interaction terms:")
            report_lines.append(f"    (each tests whether the CHANGE FROM BASELINE differs")
            report_lines.append(f"     between groups at that follow-up year, vs Year 0 reference)")
            for t in inter_terms:
                lo = ci.loc[t, 0] if t in ci.index else np.nan
                hi = ci.loc[t, 1] if t in ci.index else np.nan
                report_lines.append(f"    {t}: coef={fit.params[t]:+.3f} "
                                   f"[95%CI {lo:+.3f}, {hi:+.3f}], p={fit.pvalues[t]:.4f}")
                mixed_rows.append({"cohort": cname, "model": "primary", "term": t,
                                   "coef": round(fit.params[t],4),
                                   "ci_lo": round(lo,4), "ci_hi": round(hi,4),
                                   "p_value": round(fit.pvalues[t],4)})

            base_map = (df[df["year"] == 0]
                        .set_index("patient_id")["a1c"].to_dict())
            md_adj = md[md["year"] != 0].copy()
            md_adj["baseline_a1c"] = md_adj["patient_id"].map(base_map)
            md_adj = md_adj.dropna(subset=["baseline_a1c"])
            md_adj["year_cat"] = md_adj["year"].astype("category")
            f2 = None
            for opt in ("lbfgs", "bfgs", "cg"):
                try:
                    m2 = smf.mixedlm("a1c ~ group_bin * C(year_cat) + baseline_a1c",
                                     md_adj, groups=md_adj["patient_id"])
                    cand = m2.fit(method=opt, disp=False)
                    f2 = cand
                    break
                except Exception:
                    continue
            if f2 is not None:
                report_lines.append(f"\nSENSITIVITY model (baseline-A1c-adjusted, "
                                   f"post-baseline obs only):")
                report_lines.append(f"  group main effect p = "
                                   f"{f2.pvalues.get('group_bin', np.nan):.4f}")
                report_lines.append(f"  baseline_a1c coef = {f2.params.get('baseline_a1c', np.nan):+.3f} "
                                   f"(p={f2.pvalues.get('baseline_a1c', np.nan):.4f})")
                for t in [t for t in f2.pvalues.index if "group_bin:" in t]:
                    report_lines.append(f"    {t}: coef={f2.params[t]:+.3f}, p={f2.pvalues[t]:.4f}")
                    mixed_rows.append({"cohort": cname, "model": "baseline_adjusted", "term": t,
                                       "coef": round(f2.params[t],4),
                                       "p_value": round(f2.pvalues[t],4)})
            else:
                report_lines.append(f"\nSENSITIVITY model (baseline-adjusted): MixedLM "
                                   f"singular; using OLS with cluster-robust SE by patient")
                try:
                    import statsmodels.formula.api as smf2
                    ols = smf2.ols("a1c ~ group_bin * C(year_cat) + baseline_a1c",
                                   md_adj).fit(cov_type="cluster",
                                   cov_kwds={"groups": md_adj["patient_id"]})
                    report_lines.append(f"  group main effect p = "
                                       f"{ols.pvalues.get('group_bin', np.nan):.4f}")
                    for t in [t for t in ols.pvalues.index if "group_bin:" in t]:
                        report_lines.append(f"    {t}: coef={ols.params[t]:+.3f}, p={ols.pvalues[t]:.4f}")
                except Exception as e:
                    report_lines.append(f"  baseline-adjusted fallback also failed: {e}")
        except Exception as e:
            report_lines.append(f"\nMixed model failed: {e}")
    else:
        report_lines.append("\n(statsmodels not available — mixed model skipped)")

# coverage table
coverage_rows = []
for cname, fname in COHORTS.items():
    df = pd.read_csv(fname, dtype={"patient_id": str})
    df["a1c"] = pd.to_numeric(df["a1c"], errors="coerce")
    for grp in ["gastroparesis", "comparator"]:
        g = df[df["group"] == grp]
        eligible = g[g["year"] == 0]["patient_id"].nunique()
        for yr in sorted(df["year"].unique()):
            with_a1c = g[(g["year"] == yr) & (g["a1c"].notna())]["patient_id"].nunique()
            coverage_rows.append({
                "cohort": cname, "group": grp, "year": yr,
                "eligible_patients": eligible,
                "patients_with_a1c": with_a1c,
                "percent_available": round(100*with_a1c/eligible, 1) if eligible else 0,
            })
pd.DataFrame(coverage_rows).to_csv("a1c_stats_coverage_optimal.csv", index=False)

pd.DataFrame(timepoint_rows).to_csv("a1c_stats_timepoint_optimal.csv", index=False)
if mixed_rows:
    pd.DataFrame(mixed_rows).to_csv("a1c_stats_mixedmodel_optimal.csv", index=False)

report = "\n".join(report_lines)
with open("a1c_stats_summary_optimal.txt", "w") as f:
    f.write(report)
print(report)
print("\n\nWrote: a1c_stats_timepoint_optimal.csv, a1c_stats_mixedmodel_optimal.csv, "
      "a1c_stats_coverage_optimal.csv, a1c_stats_summary_optimal.txt")
