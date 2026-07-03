"""
balance_reconciliation.py

Purpose: reconcile the heart_failure-vs-hispanic discrepancy and expose whether
the "which covariate is imbalanced" difference is real or just borderline
covariates wobbling across the 0.1 SMD threshold.

Two prior scripts disagreed on the unpenalized run's lone imbalanced covariate:
  - paper_replication.py : unpenalized on RAW (unscaled) covariates -> heart_failure
  - isolate_grid.py      : unpenalized on SCALED covariates          -> ethnicity_hispanic
Root cause: StandardScaler on/off changes the PS model, hence the pairs.

This script runs optimal matching (no caliper) under a FULL factorial:
    scaling = {scaled, raw}
    model   = {L2, unpenalized}
    distance= {logit, raw_ps}
and for EACH cell prints the ACTUAL SMD of every covariate that is within
0.05 of the 0.1 threshold (i.e. 0.05-0.15), so we can SEE the wobble instead
of trusting a binary count. Full per-covariate SMD tables are written to CSV.

Reads only existing files. Installs nothing.
Outputs:
    balance_reconciliation.txt
    balance_reconciliation_full_smd.csv
"""

import warnings, itertools
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

df = pd.read_csv(INPUT_MATRIX).dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
y = df[TREAT_COL].astype(int).values
X_raw = df[COVARIATES].values.astype(float)
X_scaled = StandardScaler().fit_transform(X_raw)

greedy = pd.read_csv(GREEDY_PAIRS)
g_map = dict(zip(greedy["gp_patient_id"].astype(str), greedy["comp_patient_id"].astype(str)))

def run(scaling, model, distance):
    X = X_scaled if scaling == "scaled" else X_raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = (LogisticRegression(max_iter=1000, random_state=42) if model == "L2"
              else LogisticRegression(penalty=None, max_iter=5000, random_state=42))
        lr.fit(X, y)
    ps = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    d = df.copy()
    d["ps"] = ps
    d["logit"] = np.log(ps / (1 - ps))
    metric = "logit" if distance == "logit" else "ps"
    gp   = d[d[TREAT_COL] == 1].reset_index(drop=True)
    comp = d[d[TREAT_COL] == 0].reset_index(drop=True)
    cost = np.abs(gp[metric].values[:, None] - comp[metric].values[None, :])
    r, c = linear_sum_assignment(cost)
    gp_m, comp_m = gp.loc[r].reset_index(drop=True), comp.loc[c].reset_index(drop=True)
    smds = {cov: abs(smd(gp_m[cov], comp_m[cov])) for cov in COVARIATES}
    o_map = dict(zip(gp.loc[r, ID_COL].astype(str), comp.loc[c, ID_COL].astype(str)))
    shared = set(g_map) & set(o_map)
    churn = sum(1 for g in shared if g_map[g] != o_map[g])
    return smds, churn

cells = list(itertools.product(["scaled", "raw"], ["L2", "unpen"], ["logit", "raw_ps"]))
all_smds = {}
churns = {}
for cell in cells:
    smds, churn = run(*cell)
    all_smds[cell] = smds
    churns[cell] = churn

# full SMD table to CSV
full = pd.DataFrame({f"{s}|{m}|{d}": all_smds[(s,m,d)] for (s,m,d) in cells})
full.index.name = "covariate"
full.to_csv("balance_reconciliation_full_smd.csv")

with open("balance_reconciliation.txt", "w") as f:
    f.write("BALANCE RECONCILIATION — full factorial, actual SMD values\n")
    f.write("="*70 + "\n\n")
    f.write("Greedy reference (L2, scaled, logit, +caliper): 28/31 balanced,\n")
    f.write("  off on: preoperative_bmi, cad, rapid_insulin\n\n")

    f.write(f"{'scaling':<9}{'model':<7}{'dist':<8}{'balanced':>10}{'churn':>8}   lone/few imbalanced\n")
    f.write("-"*78 + "\n")
    for cell in cells:
        s, m, d = cell
        smds = all_smds[cell]
        nbal = sum(v < 0.1 for v in smds.values())
        imb = [f"{k}={smds[k]:.3f}" for k in COVARIATES if smds[k] >= 0.1]
        f.write(f"{s:<9}{m:<7}{d:<8}{nbal:>7}/31{churns[cell]:>8}   "
                f"{', '.join(imb) if imb else 'none'}\n")

    f.write("\n" + "="*70 + "\n")
    f.write("THRESHOLD WOBBLE CHECK\n")
    f.write("For the key covariates, SMD across all 8 cells.\n")
    f.write("If a covariate hovers near 0.1 and crosses it between cells, the\n")
    f.write("'which is imbalanced' difference is wobble, not a real effect.\n\n")
    watch = ["preoperative_bmi", "cad", "rapid_insulin", "heart_failure",
             "ethnicity_hispanic", "dm_renal", "sleeve_vs_bypass"]
    header = "covariate".ljust(22) + "".join(f"{s[:3]}/{m[:3]}/{d[:3]}".rjust(14)
             for (s,m,d) in cells)
    f.write(header + "\n")
    for cov in watch:
        row = cov.ljust(22) + "".join(f"{all_smds[cell][cov]:>14.3f}" for cell in cells)
        f.write(row + "\n")
    f.write("\n(threshold = 0.100; values above it are 'imbalanced')\n")

print(open("balance_reconciliation.txt").read())
