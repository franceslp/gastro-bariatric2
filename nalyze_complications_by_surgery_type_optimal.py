#!/usr/bin/env python3
"""
analyze_complications_by_surgery_type_optimal.py

Stratifies new-onset diabetes complications by surgery type (sleeve vs bypass),
mirroring analyze_ed_hosp_by_surgery_type_optimal.py.

Two analyses, both EXPLORATORY / hypothesis-generating (cells are thin):
  A) Sleeve vs Bypass within GP patients
  B) GP vs Comparator within each surgery type

New-onset definition: patient must be free of that complication at baseline
(same as the main complications analysis). Denominators therefore vary by row.

Empty subgroups (n=0) report OR as "NE (empty group)". Zero-cell tables
(a complication with no events in one arm) use the Haldane-Anscombe continuity
correction and are marked with '*'. ORs are reported with 95% CIs.

Reads:
  dm_complications_binary_with_BMI_optimal.csv
  psm_full_covariate_matrix.csv   (for sleeve_vs_bypass: 1=sleeve, 0=bypass)
Writes:
  complications_surgtype_summary_optimal.txt
  complications_surgtype_results_optimal.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

COMPLICATIONS = ["ckd", "esrd", "kidney_transplant", "heart_failure", "mi",
                 "stroke", "retinopathy", "glaucoma", "hyperglycemic_crisis",
                 "any_complication"]
LABELS = {"ckd": "CKD", "esrd": "ESRD", "kidney_transplant": "Kidney transplant",
          "heart_failure": "Heart failure", "mi": "MI", "stroke": "Stroke",
          "retinopathy": "Retinopathy", "glaucoma": "Glaucoma",
          "hyperglycemic_crisis": "Hyperglycemic crisis",
          "any_complication": "Any complication"}

df = pd.read_csv("dm_complications_binary_with_BMI_optimal.csv", dtype={"patient_id": str})
mat = pd.read_csv("psm_full_covariate_matrix.csv", dtype={"patient_id": str})

# merge surgery type
st = mat[["patient_id", "sleeve_vs_bypass"]].copy()
df = df.merge(st, on="patient_id", how="left")
n_missing_st = df["sleeve_vs_bypass"].isna().sum()
assert n_missing_st == 0, f"{n_missing_st} patients missing surgery type"
# validate coding is strictly 0/1, then map explicitly (don't treat non-1 as bypass)
assert set(df["sleeve_vs_bypass"].unique()) <= {0, 1}, \
    f"unexpected surgery coding: {set(df['sleeve_vs_bypass'].unique())}"
df["surgery"] = df["sleeve_vs_bypass"].map({1: "sleeve", 0: "bypass"})
assert df["surgery"].notna().all(), "surgery mapping produced NaN"

def or_fisher(gp_e, gp_n, co_e, co_n):
    """Return (OR_string_with_CI, p-value).

    Empty groups (n=0) return "NE (empty group)".
    Zero-cell tables use the Haldane-Anscombe continuity correction, marked '*'.
    """
    if gp_n == 0 or co_n == 0:
        return "NE (empty group)", np.nan
    a, b, c, d = gp_e, gp_n - gp_e, co_e, co_n - co_e
    try:
        _, p = stats.fisher_exact([[a, b], [c, d]])
    except Exception:
        p = np.nan
    corrected = min(a, b, c, d) == 0
    if corrected:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    OR = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo, hi = np.exp(np.log(OR) - 1.96 * se), np.exp(np.log(OR) + 1.96 * se)
    star = "*" if corrected else ""
    return f"{OR:.2f} ({lo:.2f}-{hi:.2f}){star}", p

report = []
rows = []
report.append("=" * 70)
report.append("DIABETES COMPLICATIONS BY SURGERY TYPE (Optimal Cohort)")
report.append("New-onset only (baseline-free denominators). 5-year window.")
report.append("EXPLORATORY / hypothesis-generating -- cells thin, many non-significant.")
report.append("OR shown with 95% CI. 'NE' = not estimable (empty subgroup).")
report.append("* = Haldane-Anscombe correction applied (>=1 cell had zero events).")
report.append("=" * 70)

gp = df[df.group == "gastroparesis"]
co = df[df.group == "comparator"]

# ---------- Analysis A: Sleeve vs Bypass, GP patients only ----------
report.append("\n[Analysis A - Sleeve vs Bypass, GP patients only]")
report.append(f"  GP sleeve n={(gp.surgery=='sleeve').sum()}, "
              f"GP bypass n={(gp.surgery=='bypass').sum()}")
report.append(f"  {'Complication':<22}{'Sleeve evt/n':>14}{'Bypass evt/n':>14}{'OR':>24}{'p':>9}")
for comp in COMPLICATIONS:
    base, evt = f"{comp}_baseline", f"{comp}_new_5yr"
    sl = gp[(gp.surgery == "sleeve") & (gp[base] == 0)]
    by = gp[(gp.surgery == "bypass") & (gp[base] == 0)]
    sl_e, sl_n = int((sl[evt] == 1).sum()), len(sl)
    by_e, by_n = int((by[evt] == 1).sum()), len(by)
    or_str, p = or_fisher(sl_e, sl_n, by_e, by_n)
    pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
    report.append(f"  {LABELS[comp]:<22}{f'{sl_e}/{sl_n}':>14}{f'{by_e}/{by_n}':>14}{or_str:>24}{pstr:>9}")
    rows.append({"analysis": "A_sleeve_vs_bypass_GP", "complication": LABELS[comp],
                 "sleeve_events": sl_e, "sleeve_n": sl_n,
                 "bypass_events": by_e, "bypass_n": by_n, "OR": or_str, "p": pstr})

# ---------- Analysis B: GP vs Comparator, within each surgery type ----------
report.append("\n[Analysis B - GP vs Comparator, within each surgery type]")
report.append("  NOTE: subgroup cells thin; treat as hypothesis-generating only")
for surg in ["sleeve", "bypass"]:
    g = gp[gp.surgery == surg]
    c = co[co.surgery == surg]
    report.append(f"\n  {surg.upper()} - GP (n={len(g)}) vs Comp (n={len(c)}):")
    report.append(f"    {'Complication':<22}{'GP evt/n':>12}{'Comp evt/n':>12}{'OR':>24}{'p':>9}")
    for comp in COMPLICATIONS:
        base, evt = f"{comp}_baseline", f"{comp}_new_5yr"
        gg = g[g[base] == 0]
        cc = c[c[base] == 0]
        g_e, g_n = int((gg[evt] == 1).sum()), len(gg)
        c_e, c_n = int((cc[evt] == 1).sum()), len(cc)
        or_str, p = or_fisher(g_e, g_n, c_e, c_n)
        pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
        report.append(f"    {LABELS[comp]:<22}{f'{g_e}/{g_n}':>12}{f'{c_e}/{c_n}':>12}{or_str:>24}{pstr:>9}")
        rows.append({"analysis": f"B_GP_vs_Comp_{surg}", "complication": LABELS[comp],
                     "gp_events": g_e, "gp_n": g_n, "comp_events": c_e, "comp_n": c_n,
                     "OR": or_str, "p": pstr})

text = "\n".join(report)
with open("complications_surgtype_summary_optimal.txt", "w") as f:
    f.write(text)
pd.DataFrame(rows).to_csv("complications_surgtype_results_optimal.csv", index=False)
print(text)
print("\nWrote: complications_surgtype_summary_optimal.txt, complications_surgtype_results_optimal.csv")
