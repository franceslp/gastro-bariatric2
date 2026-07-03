"""
isolate_churn_and_balance.py

Answers two questions by isolating each ingredient of the paper's method:

  Q1 (churn): the full paper run reshuffled 201/222 patients, but algorithm-only
      reshuffled just 35/222 and the L2-vs-unpenalized models were near-identical.
      So what actually drives the churn — the distance SCALE (logit vs raw PS)?

  Q2 (balance): greedy = 28/31, algorithm-only optimal = 26/31 (worse!),
      paper optimal = 30/31 (better). Which ingredient improves balance?

Method: run OPTIMAL matching (no caliper) under every combination of
        model  = {L2-penalized, unpenalized}
        scale  = {logit, raw PS}
and report, for each: matched pairs, balance (n/31 SMD<0.1), and churn vs greedy.

This is a 2x2 factorial so we can read off which lever moves what.

Reads only existing files; installs nothing.
Output: isolation_grid.txt
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

INPUT_MATRIX = "psm_full_covariate_matrix.csv"
GREEDY_PAIRS = "psm_matched_pairs_new.csv"
ID_COL, TREAT_COL = "patient_id", "group_encoded"

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

def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled

# load + complete case
df = pd.read_csv(INPUT_MATRIX).dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
y = df[TREAT_COL].astype(int).values
X = df[COVARIATES].values
Xs = StandardScaler().fit_transform(X)

def ps_for(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = (LogisticRegression(max_iter=1000, random_state=42) if model == "L2"
              else LogisticRegression(penalty=None, max_iter=5000, random_state=42))
        lr.fit(Xs, y)
    return np.clip(lr.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)

# greedy reference for churn
greedy = pd.read_csv(GREEDY_PAIRS)
g_map = dict(zip(greedy["gp_patient_id"].astype(str), greedy["comp_patient_id"].astype(str)))
greedy_gp = set(g_map)

def run_combo(model, scale):
    ps = ps_for(model)
    d = df.copy()
    d["ps"] = ps
    d["logit"] = np.log(ps / (1 - ps))
    metric = "logit" if scale == "logit" else "ps"
    gp   = d[d[TREAT_COL] == 1].reset_index(drop=True)
    comp = d[d[TREAT_COL] == 0].reset_index(drop=True)
    cost = np.abs(gp[metric].values[:, None] - comp[metric].values[None, :])
    r, c = linear_sum_assignment(cost)
    gp_m, comp_m = gp.loc[r].reset_index(drop=True), comp.loc[c].reset_index(drop=True)
    n_bal = sum(abs(smd(gp_m[cov], comp_m[cov])) < 0.1 for cov in COVARIATES)
    # churn vs greedy on shared GP
    o_map = dict(zip(gp.loc[r, ID_COL].astype(str), comp.loc[c, ID_COL].astype(str)))
    shared = greedy_gp & set(o_map)
    diff = sum(1 for g in shared if g_map[g] != o_map[g])
    # which covariates remain imbalanced
    imb = [cov for cov in COVARIATES if abs(smd(gp_m[cov], comp_m[cov])) >= 0.1]
    return len(gp_m), n_bal, diff, len(shared), imb

combos = [("L2", "logit"), ("L2", "raw"), ("unpen", "logit"), ("unpen", "raw")]
results = {}
for m, s in combos:
    results[(m, s)] = run_combo(m, s)

with open("isolation_grid.txt", "w") as f:
    f.write("2x2 ISOLATION: optimal matching (no caliper) under model x scale\n")
    f.write("="*70 + "\n\n")
    f.write("Reference points:\n")
    f.write("  greedy (L2, logit, +caliper):  222 pairs, 28/31 balanced\n\n")
    f.write(f"{'model':<8}{'scale':<8}{'pairs':>7}{'balance':>10}{'churn vs greedy':>18}\n")
    f.write("-"*51 + "\n")
    for (m, s) in combos:
        pairs, nbal, diff, shared, imb = results[(m, s)]
        f.write(f"{m:<8}{s:<8}{pairs:>7}{nbal:>7}/31"
                f"{diff:>10}/{shared}\n")
    f.write("\n")

    # highlight the two you already ran
    f.write("Cross-check against earlier runs:\n")
    f.write("  (L2, logit)  should ~match algorithm-only: 26/31, churn 35\n")
    f.write("  (unpen, raw) should ~match paper run:       30/31, churn 201\n\n")

    # read off the levers
    f.write("READING THE LEVERS\n")
    f.write("-"*70 + "\n")
    b = {k: results[k][1] for k in results}
    c = {k: results[k][2] for k in results}
    f.write("Balance effect of SCALE (holding model fixed):\n")
    f.write(f"  L2:    logit {b[('L2','logit')]}/31  ->  raw {b[('L2','raw')]}/31\n")
    f.write(f"  unpen: logit {b[('unpen','logit')]}/31  ->  raw {b[('unpen','raw')]}/31\n")
    f.write("Balance effect of MODEL (holding scale fixed):\n")
    f.write(f"  logit: L2 {b[('L2','logit')]}/31  ->  unpen {b[('unpen','logit')]}/31\n")
    f.write(f"  raw:   L2 {b[('L2','raw')]}/31  ->  unpen {b[('unpen','raw')]}/31\n\n")
    f.write("Churn effect of SCALE (holding model fixed):\n")
    f.write(f"  L2:    logit {c[('L2','logit')]}  ->  raw {c[('L2','raw')]}\n")
    f.write(f"  unpen: logit {c[('unpen','logit')]}  ->  raw {c[('unpen','raw')]}\n")
    f.write("Churn effect of MODEL (holding scale fixed):\n")
    f.write(f"  logit: L2 {c[('L2','logit')]}  ->  unpen {c[('unpen','logit')]}\n")
    f.write(f"  raw:   L2 {c[('L2','raw')]}  ->  unpen {c[('unpen','raw')]}\n\n")

    # imbalanced covariates for the paper-equivalent cell
    f.write("Imbalanced covariates by cell (SMD>=0.1):\n")
    for (m, s) in combos:
        imb = results[(m, s)][4]
        f.write(f"  ({m},{s}): {', '.join(imb) if imb else 'none'}\n")

print(open("isolation_grid.txt").read())
