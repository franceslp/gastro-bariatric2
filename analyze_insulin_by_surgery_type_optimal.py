#!/usr/bin/env python3
"""
analyze_insulin_by_surgery_type_optimal.py

Stratifies new insulin initiation by surgery type (sleeve vs bypass),
mirroring analyze_ed_hosp_by_surgery_type_optimal.py.

Two analyses, both EXPLORATORY / hypothesis-generating (cells are thin):
  A) Sleeve vs Bypass within GP patients
  B) GP vs Comparator within each surgery type

Done separately for rapid-acting and long-acting insulin (baseline non-users only,
which is already how insulin_initiation_events_with_BMI_optimal.csv is built:
one row per patient per insulin_type, among baseline non-users).

Binary ever-initiation OR (Fisher exact) with 95% CI. Empty subgroups report
"NE (empty group)"; zero-cell tables use the Haldane-Anscombe correction (marked '*').
(Cox HR by surgery type omitted here: subgroup cells too thin for a stable
proportional-hazards fit; binary OR is the honest level of detail. Can add later.)

Reads:
  insulin_initiation_events_with_BMI_optimal.csv   (patient_id, group, insulin_type, event, time, had_followup)
  psm_full_covariate_matrix.csv                    (sleeve_vs_bypass: 1=sleeve, 0=bypass)
Writes:
  insulin_surgtype_summary_optimal.txt
  insulin_surgtype_results_optimal.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("insulin_initiation_events_with_BMI_optimal.csv", dtype={"patient_id": str})
mat = pd.read_csv("psm_full_covariate_matrix.csv", dtype={"patient_id": str})

st = mat[["patient_id", "sleeve_vs_bypass"]].copy()
df = df.merge(st, on="patient_id", how="left")
assert df["sleeve_vs_bypass"].isna().sum() == 0, "some patients missing surgery type"
# validate coding is strictly 0/1, then map explicitly (don't treat non-1 as bypass)
assert set(df["sleeve_vs_bypass"].unique()) <= {0, 1}, \
    f"unexpected surgery coding: {set(df['sleeve_vs_bypass'].unique())}"
df["surgery"] = df["sleeve_vs_bypass"].map({1: "sleeve", 0: "bypass"})
assert df["surgery"].notna().all(), "surgery mapping produced NaN"

INSULIN_TYPES = sorted(df["insulin_type"].unique())  # e.g. ['long','rapid']

def or_fisher(a_e, a_n, b_e, b_n):
    """Return (OR_str_with_CI, p). '*' marks Haldane-Anscombe correction."""
    if a_n == 0 or b_n == 0:
        return "NE (empty group)", np.nan
    a, b, c, d = a_e, a_n - a_e, b_e, b_n - b_e
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
report.append("NEW INSULIN INITIATION BY SURGERY TYPE (Optimal Cohort)")
report.append("Among baseline non-users of each insulin type. Binary ever-initiation.")
report.append("EXPLORATORY / hypothesis-generating -- cells thin, many non-significant.")
report.append("OR shown with 95% CI. 'NE' = not estimable (empty subgroup).")
report.append("* = Haldane-Anscombe correction applied (>=1 cell had zero events).")
report.append("=" * 70)

gp = df[df.group == "gastroparesis"]
co = df[df.group == "comparator"]

for itype in INSULIN_TYPES:
    report.append(f"\n{'='*70}\nINSULIN TYPE: {itype.upper()}\n{'='*70}")
    gpi = gp[gp.insulin_type == itype]
    coi = co[co.insulin_type == itype]

    # ---- Analysis A: Sleeve vs Bypass, GP only ----
    sl = gpi[gpi.surgery == "sleeve"]
    by = gpi[gpi.surgery == "bypass"]
    sl_e, sl_n = int((sl["event"] == 1).sum()), len(sl)
    by_e, by_n = int((by["event"] == 1).sum()), len(by)
    or_str, p = or_fisher(sl_e, sl_n, by_e, by_n)
    pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
    report.append(f"\n[Analysis A - Sleeve vs Bypass, GP patients only]")
    report.append(f"  Sleeve {sl_e}/{sl_n} vs Bypass {by_e}/{by_n}  OR={or_str}, p={pstr}")
    rows.append({"insulin_type": itype, "analysis": "A_sleeve_vs_bypass_GP",
                 "sleeve_events": sl_e, "sleeve_n": sl_n,
                 "bypass_events": by_e, "bypass_n": by_n, "OR": or_str, "p": pstr})

    # ---- Analysis B: GP vs Comparator, within each surgery type ----
    report.append(f"\n[Analysis B - GP vs Comparator, within each surgery type]")
    for surg in ["sleeve", "bypass"]:
        g = gpi[gpi.surgery == surg]
        c = coi[coi.surgery == surg]
        g_e, g_n = int((g["event"] == 1).sum()), len(g)
        c_e, c_n = int((c["event"] == 1).sum()), len(c)
        or_str, p = or_fisher(g_e, g_n, c_e, c_n)
        pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
        report.append(f"  {surg.upper()}: GP {g_e}/{g_n} vs Comp {c_e}/{c_n}  OR={or_str}, p={pstr}")
        rows.append({"insulin_type": itype, "analysis": f"B_GP_vs_Comp_{surg}",
                     "gp_events": g_e, "gp_n": g_n, "comp_events": c_e, "comp_n": c_n,
                     "OR": or_str, "p": pstr})

text = "\n".join(report)
with open("insulin_surgtype_summary_optimal.txt", "w") as f:
    f.write(text)
pd.DataFrame(rows).to_csv("insulin_surgtype_results_optimal.csv", index=False)
print(text)
print("\nWrote: insulin_surgtype_summary_optimal.txt, insulin_surgtype_results_optimal.csv")
