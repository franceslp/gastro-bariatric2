#!/usr/bin/env python3
"""
analyze_complications_5yr_strict_by_surgery_type_optimal.py

Retention-restricted (true 5-year follow-up) complications, split by surgery
type -- the surgery-type analog of the "restricted to patients with full
5-year follow-up" table built for the whole cohort in
compute_5yr_restricted.py.

Denominator = baseline-free AND actually followed >=1825 days (per
followup_days_post from the ED/hosp file, same follow-up definition used
throughout this project). Same two analyses as the available-case version:
  A) Sleeve vs Bypass within GP patients
  B) GP vs Comparator within each surgery type

EXPLORATORY / hypothesis-generating -- expect very thin cells (the whole-
cohort retention-restricted table was already down to ~27-47 at-risk
patients for some rows; splitting further by surgery type will push several
cells to single digits or zero). This is a standalone, separate script from
analyze_complications_by_surgery_type_optimal.py (the available-case version)
so as not to touch or re-run that already-validated script.

Reads:
  dm_complications_binary_with_BMI_optimal.csv
  ed_hosp_binary_with_BMI_optimal.csv   (for followup_days_post)
  psm_full_covariate_matrix.csv         (for sleeve_vs_bypass: 1=sleeve, 0=bypass)
Writes:
  complications_5yr_strict_surgtype_summary_optimal.txt
  complications_5yr_strict_surgtype_results_optimal.csv
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

comp = pd.read_csv("dm_complications_binary_with_BMI_optimal.csv", dtype={"patient_id": str})
ed = pd.read_csv("ed_hosp_binary_with_BMI_optimal.csv", dtype={"patient_id": str})
mat = pd.read_csv("psm_full_covariate_matrix.csv", dtype={"patient_id": str})

# merge follow-up duration
fu = ed[["patient_id", "followup_days_post"]]
df = comp.merge(fu, on="patient_id", how="left")
assert df["followup_days_post"].notna().all(), "some patients missing follow-up"

# merge + validate surgery type (same guard as the available-case script)
st = mat[["patient_id", "sleeve_vs_bypass"]].copy()
df = df.merge(st, on="patient_id", how="left")
assert df["sleeve_vs_bypass"].isna().sum() == 0, "some patients missing surgery type"
assert set(df["sleeve_vs_bypass"].unique()) <= {0, 1}, \
    f"unexpected surgery coding: {set(df['sleeve_vs_bypass'].unique())}"
df["surgery"] = df["sleeve_vs_bypass"].map({1: "sleeve", 0: "bypass"})
assert df["surgery"].notna().all(), "surgery mapping produced NaN"

df["reached5"] = df["followup_days_post"] >= 1825

def or_fisher(a_e, a_n, b_e, b_n):
    """Return (OR_string_with_CI, p-value).

    Empty groups (n=0) return "NE (empty group)".
    Zero-cell tables use the Haldane-Anscombe continuity correction, marked '*'.
    """
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
report.append("=" * 72)
report.append("DIABETES COMPLICATIONS BY SURGERY TYPE -- 5-YEAR RETENTION-RESTRICTED")
report.append("(Optimal Cohort)")
report.append("Denominator = baseline-free AND >=1825 days actual follow-up.")
report.append("EXPLORATORY / hypothesis-generating -- expect very thin cells; several")
report.append("rows may be 'NE' or Haldane-corrected. DESCRIPTIVE ONLY, not a reliable")
report.append("hypothesis test at this sample size.")
report.append("OR shown with 95% CI. 'NE' = not estimable (empty subgroup).")
report.append("* = Haldane-Anscombe correction applied (>=1 cell had zero events).")
report.append("=" * 72)

gp = df[df.group == "gastroparesis"]
co = df[df.group == "comparator"]

# retention summary by surgery type (for transparency -- how thin are we?)
report.append("\n5-year retention, by group and surgery type:")
for grp_name, grp_df in [("GP", gp), ("Comparator", co)]:
    for surg in ["sleeve", "bypass"]:
        sub = grp_df[grp_df.surgery == surg]
        n_reached = int(sub["reached5"].sum())
        report.append(f"  {grp_name} {surg}: {n_reached}/{len(sub)} reached 5yr follow-up")

# ---------- Analysis A: Sleeve vs Bypass, GP patients only ----------
report.append("\n[Analysis A - Sleeve vs Bypass, GP patients only, 5yr-retention-restricted]")
report.append(f"  {'Complication':<22}{'Sleeve evt/atrisk':>18}{'Bypass evt/atrisk':>18}{'OR':>24}{'p':>9}")
for c in COMPLICATIONS:
    base, evt = f"{c}_baseline", f"{c}_new_5yr"
    sl = gp[(gp.surgery == "sleeve") & (gp[base] == 0) & (gp["reached5"])]
    by = gp[(gp.surgery == "bypass") & (gp[base] == 0) & (gp["reached5"])]
    sl_e, sl_n = int((sl[evt] == 1).sum()), len(sl)
    by_e, by_n = int((by[evt] == 1).sum()), len(by)
    or_str, p = or_fisher(sl_e, sl_n, by_e, by_n)
    pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
    report.append(f"  {LABELS[c]:<22}{f'{sl_e}/{sl_n}':>18}{f'{by_e}/{by_n}':>18}{or_str:>24}{pstr:>9}")
    rows.append({"analysis": "A_sleeve_vs_bypass_GP_5yr_strict", "complication": LABELS[c],
                 "sleeve_events": sl_e, "sleeve_atrisk": sl_n,
                 "bypass_events": by_e, "bypass_atrisk": by_n, "OR": or_str, "p": pstr})

# ---------- Analysis B: GP vs Comparator, within each surgery type ----------
report.append("\n[Analysis B - GP vs Comparator, within each surgery type, 5yr-retention-restricted]")
report.append("  NOTE: cells very thin; descriptive only")
for surg in ["sleeve", "bypass"]:
    g_all = gp[(gp.surgery == surg) & (gp["reached5"])]
    c_all = co[(co.surgery == surg) & (co["reached5"])]
    report.append(f"\n  {surg.upper()} - GP (n={len(g_all)}) vs Comp (n={len(c_all)}) [5yr-retention-restricted]:")
    report.append(f"    {'Complication':<22}{'GP evt/atrisk':>16}{'Comp evt/atrisk':>16}{'OR':>24}{'p':>9}")
    for c in COMPLICATIONS:
        base, evt = f"{c}_baseline", f"{c}_new_5yr"
        gg = gp[(gp.surgery == surg) & (gp["reached5"]) & (gp[base] == 0)]
        cc = co[(co.surgery == surg) & (co["reached5"]) & (co[base] == 0)]
        g_e, g_n = int((gg[evt] == 1).sum()), len(gg)
        c_e, c_n = int((cc[evt] == 1).sum()), len(cc)
        or_str, p = or_fisher(g_e, g_n, c_e, c_n)
        pstr = f"{p:.4f}" if not np.isnan(p) else "NA"
        report.append(f"    {LABELS[c]:<22}{f'{g_e}/{g_n}':>16}{f'{c_e}/{c_n}':>16}{or_str:>24}{pstr:>9}")
        rows.append({"analysis": f"B_GP_vs_Comp_{surg}_5yr_strict", "complication": LABELS[c],
                     "gp_events": g_e, "gp_atrisk": g_n, "comp_events": c_e, "comp_atrisk": c_n,
                     "OR": or_str, "p": pstr})

text = "\n".join(report)
with open("complications_5yr_strict_surgtype_summary_optimal.txt", "w") as f:
    f.write(text)
pd.DataFrame(rows).to_csv("complications_5yr_strict_surgtype_results_optimal.csv", index=False)
print(text)
print("\nWrote: complications_5yr_strict_surgtype_summary_optimal.txt, "
      "complications_5yr_strict_surgtype_results_optimal.csv")
