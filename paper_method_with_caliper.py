"""
paper_method_with_caliper.py

Tests a gap we never filled: the paper's optimal method (unpenalized glm,
raw inputs, PS-scale distance, optimal matching) was only ever run WITHOUT a
caliper (30/31, lone miss = heart_failure). Question: if we add your standard
0.2 SD logit caliper to it, does it still match everyone, and does the lone
imbalanced covariate change?

How a caliper works with optimal matching:
  Any candidate pair whose logit-PS distance exceeds the caliper is FORBIDDEN
  (assigned infinite cost), so the Hungarian solver never selects it. Treated
  patients with no comparator inside the caliper are dropped. This mirrors how
  your greedy caliper drops unmatchable patients.

Caliper scale note: the caliper is defined on the LOGIT scale (0.2 SD of
logit PS), per Austin 2011 — the standard definition — even though the paper's
matching *distance* is on the raw PS scale. The caliper is a standardized
gate; matching distance is a separate quantity.

For comparison it also runs the paper method WITHOUT caliper (to reproduce the
known 30/31 / heart_failure result as a sanity check).

Reads only existing files. Installs nothing.
Output: paper_method_with_caliper_report.txt
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment

INPUT_MATRIX = "psm_full_covariate_matrix.csv"
ID_COL, TREAT_COL = "patient_id", "group_encoded"
CALIPER = 0.2   # SD of logit PS, Austin 2011

COVARIATES = [
    "age_at_surgery_approx", "baseline_a1c", "preoperative_bmi",
    "diabetes_duration_log1p",
    "t1dm", "t2dm",
    "dm_renal", "dm_neuro", "dm_circ", "dm_opthal", "dm_other",
    "hypertension", "ckd", "cad", "stroke", "heart_failure", "dyslipidemia",
    "metformin", "any_insulin", "rapid_insulin", "long_insulin",
    "glp1", "sglt2", "dpp4", "sulfonylurea", "tzd",
    "sex_encoded", "race_white", "race_black", "ethnicity_hispanic",
    "sleeve_vs_bypass",
]
PRIORITY = ["preoperative_bmi", "any_insulin", "rapid_insulin", "long_insulin",
            "t2dm", "heart_failure"]

def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled

df = pd.read_csv(INPUT_MATRIX).dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
y = df[TREAT_COL].astype(int).values

# ---- paper's PS model: unpenalized, RAW inputs (no scaler) ----
X = df[COVARIATES].values.astype(float)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = LogisticRegression(penalty=None, max_iter=5000, random_state=42)
    lr.fit(X, y)
ps = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
df["ps"] = ps
df["logit_ps"] = np.log(ps / (1 - ps))

logit_sd = df["logit_ps"].std()
caliper_val = CALIPER * logit_sd   # gate on logit scale

gp   = df[df[TREAT_COL] == 1].reset_index(drop=True)
comp = df[df[TREAT_COL] == 0].reset_index(drop=True)

caliper_logit = CALIPER * df["logit_ps"].std()   # Austin-standard (logit scale)
caliper_rawps = CALIPER * df["ps"].std()          # raw-PS scale (paper's distance scale)

def optimal_match(caliper_scale):
    """caliper_scale in {None,'logit','rawps'}; matching distance always raw PS."""
    ps_cost = np.abs(gp["ps"].values[:, None] - comp["ps"].values[None, :])
    if caliper_scale is None:
        r, c = linear_sum_assignment(ps_cost)
        return _finish(list(r), list(c))
    if caliper_scale == "logit":
        gate = np.abs(gp["logit_ps"].values[:, None] - comp["logit_ps"].values[None, :])
        thr = caliper_logit
    else:
        gate = ps_cost
        thr = caliper_rawps
    cost = ps_cost.copy()
    cost[gate > thr] = np.inf
    eligible = ~np.all(np.isinf(cost), axis=1)
    gp_idx = np.where(eligible)[0]
    cost = cost[eligible]
    BIG = 1e6
    cost_solve = np.where(np.isinf(cost), BIG, cost)
    r, c = linear_sum_assignment(cost_solve)
    keep = [(gp_idx[i], c[k]) for k, i in enumerate(r) if cost[i, c[k]] < BIG]
    return _finish([k[0] for k in keep], [k[1] for k in keep])

def _finish(gp_sel, comp_sel):
    gp_m   = gp.loc[gp_sel].reset_index(drop=True)
    comp_m = comp.loc[comp_sel].reset_index(drop=True)
    smds = {cov: abs(smd(gp_m[cov], comp_m[cov])) for cov in COVARIATES}
    return len(gp_m), smds

n_nc, smd_nc = optimal_match(None)
n_rp, smd_rp = optimal_match("rawps")   # what you asked for: caliper on raw PS
n_lg, smd_lg = optimal_match("logit")   # Austin-standard, shown alongside
n_c,  smd_c  = n_rp, smd_rp              # raw-PS is the requested primary

with open("paper_method_with_caliper_report.txt", "w") as f:
    f.write("PAPER'S OPTIMAL METHOD + caliper (raw-PS scale, as requested)\n")
    f.write("(unpenalized glm, raw inputs, PS-scale matching distance, optimal)\n")
    f.write("="*68 + "\n\n")
    f.write("NOTE: the paper used NO caliper. Adding one is an extension.\n")
    f.write("Raw-PS caliper is NOT the Austin standard (which is logit-scale);\n")
    f.write("the logit column is shown alongside for reference.\n\n")
    f.write(f"caliper thresholds:  raw-PS = {caliper_rawps:.5f}   "
            f"logit = {caliper_logit:.4f}\n\n")

    def col(smds):
        return sum(v < 0.1 for v in smds.values())
    f.write(f"{'':<24}{'no caliper':>13}{'raw-PS cal':>13}{'logit cal':>13}\n")
    f.write(f"{'pairs matched':<24}{n_nc:>13}{n_rp:>13}{n_lg:>13}\n")
    f.write(f"{'covariates <0.1':<24}{str(col(smd_nc))+'/31':>13}"
            f"{str(col(smd_rp))+'/31':>13}{str(col(smd_lg))+'/31':>13}\n\n")

    f.write("Imbalanced covariates (SMD >= 0.1):\n")
    for label, smds in [("no caliper ", smd_nc), ("raw-PS cal ", smd_rp),
                        ("logit cal  ", smd_lg)]:
        imb = {k: smds[k] for k in COVARIATES if smds[k] >= 0.1}
        f.write(f"  {label}: {', '.join(f'{k}={v:.3f}' for k,v in imb.items()) or 'none'}\n")
    f.write("\n")

    f.write("Priority / watched covariates:\n")
    f.write(f"{'covariate':<22}{'no cal':>9}{'raw-PS':>9}{'logit':>9}\n")
    for cov in PRIORITY:
        f.write(f"{cov:<22}{smd_nc[cov]:>9.3f}{smd_rp[cov]:>9.3f}{smd_lg[cov]:>9.3f}\n")
    f.write("\nSanity check: no-caliper should reproduce the known paper result\n")
    f.write("(30/31, lone miss = heart_failure ~0.144).\n")

print(open("paper_method_with_caliper_report.txt").read())
