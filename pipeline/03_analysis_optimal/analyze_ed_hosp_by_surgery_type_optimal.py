#!/usr/bin/env python3
"""
analyze_ed_hosp_by_surgery_type_optimal.py

Sleeve vs bypass comparison of ED visits and hospitalizations among GP patients.
No new GCS scan needed — joins surgery type onto existing binary output files.

ANALYSES:
  A. Sleeve vs bypass in GP patients only (matched subset):
     - Pre/post change by surgery type (Wilcoxon paired)
     - Difference-in-differences: did sleeve/bypass differ in their reduction?
     - Binary 5yr: did sleeve/bypass GP patients differ in post-op utilization?

  B. Brief stratified check (exploratory, flagged as underpowered):
     Within sleeve: GP vs comparator
     Within bypass: GP vs comparator

Uses psm_full_covariate_matrix.csv for sleeve_vs_bypass (1=sleeve, 0=bypass).
Runs on both matched cohorts (with_BMI, no_BMI).

OUTPUTS:
  ed_hosp_surgtype_summary.txt
  ed_hosp_surgtype_results.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

COHORTS = {
    "with_BMI_optimal": "ed_hosp_binary_with_BMI_optimal.csv",
}

# Load surgery type from covariate matrix
mat = pd.read_csv("psm_full_covariate_matrix.csv", dtype={"patient_id": str})
mat["sleeve_vs_bypass"] = pd.to_numeric(mat["sleeve_vs_bypass"], errors="coerce")
surg_map = {r["patient_id"]: ("sleeve" if r["sleeve_vs_bypass"]==1 else
                               "bypass" if r["sleeve_vs_bypass"]==0 else None)
            for _, r in mat.iterrows()}

report = []
report.append("="*68)
report.append("ED VISITS & HOSPITALIZATIONS BY SURGERY TYPE — GP Patients")
report.append("Sleeve vs bypass comparison using existing binary output files.")
report.append("Surgery type from psm_full_covariate_matrix.csv (1=sleeve, 0=bypass)")
report.append("="*68)

result_rows = []

def wilcoxon_pre_post(pre, post, label):
    """Paired Wilcoxon for pre->post change."""
    if len(pre) < 10:
        return f"  {label}: n too small (n={len(pre)})"
    try:
        _, p = stats.wilcoxon(post, pre, zero_method="wilcox")
    except Exception:
        p = np.nan
    delta = post.mean() - pre.mean()
    return (f"  {label}: pre {pre.mean():.2f} -> post {post.mean():.2f} "
            f"(Δ={delta:+.2f}, Wilcoxon p={p:.4f})")

def welch_diff(a, b, label_a, label_b):
    """Welch t-test comparing two groups."""
    if len(a) < 5 or len(b) < 5:
        return f"  {label_a} vs {label_b}: n too small"
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean()-b.mean()) / np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())
                                       /(len(a)+len(b)-2)) if a.var()+b.var()>0 else np.nan
    return (f"  {label_a} Δ={a.mean():+.2f} vs {label_b} Δ={b.mean():+.2f}, "
            f"p={p:.4f}, d={d:+.2f}")

def binary_or(gp_n, gp_N, co_n, co_N):
    a,b,c,d = gp_n, gp_N-gp_n, co_n, co_N-co_n
    if min(a,b,c,d)==0:
        a,b,c,d = a+.5,b+.5,c+.5,d+.5
    tbl = np.array([[a,b],[c,d]])
    try:
        _, p, _, _ = stats.chi2_contingency(tbl)
    except Exception:
        p = np.nan
    OR = (a*d)/(b*c)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    lo, hi = np.exp(np.log(OR)-1.96*se), np.exp(np.log(OR)+1.96*se)
    return OR, lo, hi, p

for cname, fname in COHORTS.items():
    df = pd.read_csv(fname, dtype={"patient_id": str})
    df["surgery_type"] = df["patient_id"].map(surg_map)

    # QA: check merge
    missing_st = df["surgery_type"].isna().sum()
    report.append(f"\n{'='*60}\nCOHORT: {cname}\n{'='*60}")
    report.append(f"  Total patients: {len(df)}, missing surgery type: {missing_st}")

    gp_df = df[df["group"] == "gastroparesis"].copy()
    comp_df = df[df["group"] == "comparator"].copy()

    # GP surgery type breakdown
    gp_sl = gp_df[gp_df["surgery_type"]=="sleeve"]
    gp_by = gp_df[gp_df["surgery_type"]=="bypass"]
    report.append(f"  GP sleeve n={len(gp_sl)}, GP bypass n={len(gp_by)}")
    report.append(f"  (Exploratory; cells thin by Year 4-5)")

    # ---- ANALYSIS A: Sleeve vs Bypass within GP patients ----
    report.append(f"\n[Analysis A — Sleeve vs Bypass, GP patients only, {cname}]")

    for pre_col, post_col, change_col, lab in [
        ("ED_pre_5yr",    "ED_post_5yr",    "ED_change",    "ED visits"),
        ("IP_pre_5yr",    "IP_post_5yr",    "IP_change",    "Hospitalizations"),
        ("acute_pre_5yr", "acute_post_5yr", "acute_change", "Acute care events"),
    ]:
        report.append(f"\n  {lab}:")
        # Pre/post within sleeve GP
        report.append(wilcoxon_pre_post(
            gp_sl[pre_col], gp_sl[post_col], f"Sleeve (n={len(gp_sl)})"))
        # Pre/post within bypass GP
        report.append(wilcoxon_pre_post(
            gp_by[pre_col], gp_by[post_col], f"Bypass (n={len(gp_by)})"))
        # DID: did the reduction differ between sleeve and bypass GP?
        report.append("  Sleeve vs Bypass (diff-in-diff: was reduction different?):")
        report.append("  " + welch_diff(
            gp_sl[change_col], gp_by[change_col], "Sleeve", "Bypass"))
        # Binary post-op 5yr
        sl_post = gp_sl[f"had_{pre_col.split('_')[0]}_post"] if f"had_{pre_col.split('_')[0]}_post" in gp_sl.columns else (gp_sl[post_col]>0)
        by_post = gp_by[f"had_{pre_col.split('_')[0]}_post"] if f"had_{pre_col.split('_')[0]}_post" in gp_by.columns else (gp_by[post_col]>0)
        # Use correct binary col name
        binary_col = ("had_ED_post" if "ED" in lab and "Acute" not in lab
                      else "had_IP_post" if "Hosp" in lab
                      else "had_acute_post")
        if binary_col in gp_sl.columns:
            sl_b = gp_sl[binary_col]; by_b = gp_by[binary_col]
            OR, lo, hi, p = binary_or(sl_b.sum(), len(sl_b), by_b.sum(), len(by_b))
            report.append(f"  Binary 5yr: Sleeve {sl_b.sum()}/{len(sl_b)} "
                         f"({100*sl_b.mean():.1f}%) vs Bypass {by_b.sum()}/{len(by_b)} "
                         f"({100*by_b.mean():.1f}%), OR={OR:.2f} "
                         f"(95%CI {lo:.2f}-{hi:.2f}), p={p:.4f}")
            result_rows.append({
                "cohort": cname, "analysis": "A_sleeve_vs_bypass_GP",
                "outcome": lab, "surgery_a": "sleeve", "surgery_b": "bypass",
                "n_a": len(sl_b), "pct_a": round(100*sl_b.mean(),1),
                "n_b": len(by_b), "pct_b": round(100*by_b.mean(),1),
                "OR": round(OR,2), "ci_lo": round(lo,2),
                "ci_hi": round(hi,2), "p": round(p,4)
            })

    # ---- ANALYSIS B: GP vs Comparator stratified by surgery type (exploratory) ----
    report.append(f"\n[Analysis B — GP vs Comparator by surgery type (exploratory)]")
    report.append(f"  NOTE: subgroup cells thin; treat as hypothesis-generating only")
    comp_sl = comp_df[comp_df["surgery_type"]=="sleeve"]
    comp_by = comp_df[comp_df["surgery_type"]=="bypass"]
    report.append(f"  Comp sleeve n={len(comp_sl)}, Comp bypass n={len(comp_by)}")

    for st, gp_sub, comp_sub in [("sleeve",gp_sl,comp_sl),("bypass",gp_by,comp_by)]:
        report.append(f"\n  {st.upper()} — GP (n={len(gp_sub)}) vs Comp (n={len(comp_sub)}):")
        for binary_col, lab in [("had_ED_post","ED visit"),
                                  ("had_IP_post","Hospitalization"),
                                  ("had_acute_post","Acute care")]:
            if binary_col not in gp_sub.columns:
                continue
            gn = gp_sub[binary_col].sum(); gN = len(gp_sub)
            cn = comp_sub[binary_col].sum(); cN = len(comp_sub)
            OR, lo, hi, p = binary_or(gn, gN, cn, cN)
            report.append(f"    {lab}: GP {gn}/{gN} ({100*gn/gN:.1f}%) vs "
                         f"Comp {cn}/{cN} ({100*cn/cN:.1f}%), "
                         f"OR={OR:.2f} ({lo:.2f}-{hi:.2f}), p={p:.4f}")

text = "\n".join(report)
with open("ed_hosp_surgtype_summary_optimal.txt", "w") as f:
    f.write(text)
pd.DataFrame(result_rows).to_csv("ed_hosp_surgtype_results_optimal.csv", index=False)
print(text)
print("\nWrote: ed_hosp_surgtype_summary_optimal.txt, ed_hosp_surgtype_results_optimal.csv")
