#!/usr/bin/env python3
"""
analyze_hosp_frequency_among_hospitalized_optimal.py

Exploratory (post hoc) analysis: among matched patients with >=1 postoperative
hospitalization (days 31-1825), compare the number of hospitalizations between
the gastroparesis and comparator groups.

Input : ed_hosp_binary_with_BMI_optimal.csv (written by
        collect_ed_hospitalizations_optimal.py)
Output: hosp_frequency_among_hospitalized_optimal.txt (new file; nothing overwritten)

Models
  1. Welch t-test on raw counts (unadjusted)
  2. Negative binomial, offset = log(follow-up years)            <- value in paper
  3. Negative binomial, offset = log(follow-up years) + pre-op hospitalizations
     (sensitivity: does the result hold after accounting for preoperative use?)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

INFILE = "ed_hosp_binary_with_BMI_optimal.csv"
OUTFILE = "hosp_frequency_among_hospitalized_optimal.txt"

lines = []
def out(s=""):
    print(s)
    lines.append(s)

df = pd.read_csv(INFILE)
df["gp"] = (df["group"].str.lower() == "gastroparesis").astype(int)
df["fu_years"] = df["followup_days_post"] / 365.25

sub = df[df["IP_post_5yr"] >= 1].copy()
gp, comp = sub[sub.gp == 1], sub[sub.gp == 0]

out("HOSPITALIZATION FREQUENCY AMONG PATIENTS HOSPITALIZED >=1x (post-op days 31-1825)")
out("Exploratory post hoc analysis, optimal matched cohort")
out("=" * 72)
out(f"n hospitalized: gastroparesis {len(gp)}, comparator {len(comp)}")
out(f"Mean hospitalizations:   GP {gp.IP_post_5yr.mean():.2f}  vs  Comp {comp.IP_post_5yr.mean():.2f}")
out(f"Median hospitalizations: GP {gp.IP_post_5yr.median():.1f}  vs  Comp {comp.IP_post_5yr.median():.1f}")
out(f"Median follow-up (yr):   GP {gp.fu_years.median():.2f}  vs  Comp {comp.fu_years.median():.2f}")
out(f"Mean pre-op hospitalizations: GP {gp.IP_pre_5yr.mean():.2f}  vs  Comp {comp.IP_pre_5yr.mean():.2f}")
out()

t = stats.ttest_ind(gp.IP_post_5yr, comp.IP_post_5yr, equal_var=False)
out(f"1. Welch t-test (unadjusted):  p = {t.pvalue:.4f}")
out()

def nb_report(label, formula, data):
    """Negative binomial with dispersion (alpha) estimated from the data."""
    off = np.log(data["fu_years"])
    m = smf.negativebinomial(formula, data=data, offset=off).fit(disp=False, maxiter=200)
    b, se = m.params["gp"], m.bse["gp"]
    lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    out(f"{label}")
    out(f"   GP IRR = {np.exp(b):.2f} (95% CI {lo:.2f}-{hi:.2f}), p = {m.pvalues['gp']:.4f}")
    if "IP_pre_5yr" in formula:
        out(f"   pre-op hospitalizations: IRR per admission = {np.exp(m.params['IP_pre_5yr']):.3f}, "
            f"p = {m.pvalues['IP_pre_5yr']:.4f}")
    out()

nb_report("2. Negative binomial, offset = log(follow-up years)  [paper reports IRR 1.90, 1.34-2.69]",
          "IP_post_5yr ~ gp", sub)
nb_report("3. SENSITIVITY: + adjustment for pre-op hospitalizations",
          "IP_post_5yr ~ gp + IP_pre_5yr", sub)

with open(OUTFILE, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote: {OUTFILE}")
