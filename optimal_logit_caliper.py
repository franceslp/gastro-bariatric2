"""
optimal_logit_caliper.py

FINAL configuration chosen for the with-BMI cohort:
  - Propensity model : UNPENALIZED logistic regression (plain glm-equivalent,
                       aligns with Kendall et al.; verified no separation)
  - Matching         : OPTIMAL 1:1 (Hungarian; minimizes total within-pair
                       distance) -- fixes greedy's arrangement problem
  - Distance scale   : LOGIT of the propensity score
  - Caliper          : 0.2 SD of the LOGIT propensity score (Austin 2011)
                       -- quality floor; drops GP patients with no comparator
                       inside the caliper
  - Balance grade    : SMD < 0.1

Reports how many patients the caliper removes, final pair count, full balance
table, and the lone/few imbalanced covariates.

Reads only existing files. Installs nothing.
Outputs:
  optimal_logit_caliper_pairs.csv
  optimal_logit_caliper_balance.csv
  optimal_logit_caliper_report.txt
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment

INPUT_MATRIX = "psm_full_covariate_matrix.csv"
ID_COL, TREAT_COL = "patient_id", "group_encoded"
CALIPER = 0.2   # SD of logit PS (Austin 2011)

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

# ---------------------------------------------------------------------------
# 1. Load + complete-case (no imputation)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_MATRIX)
before = len(df)
df = df.dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
print(f"Loaded {before}; complete cases {len(df)} (dropped {before-len(df)})")
y = df[TREAT_COL].astype(int).values
print(f"GP: {(y==1).sum()}  Comparator: {(y==0).sum()}")

# ---------------------------------------------------------------------------
# 2. Propensity score: UNPENALIZED logistic on raw covariates (glm-equivalent)
# ---------------------------------------------------------------------------
X = df[COVARIATES].values.astype(float)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = LogisticRegression(penalty=None, max_iter=5000, random_state=42)
    lr.fit(X, y)
if lr.n_iter_[0] >= 5000:
    print("WARNING: model may not have converged (possible separation).")
ps = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
df["propensity_score"] = ps
df["logit_ps"] = np.log(ps / (1 - ps))

logit_sd = df["logit_ps"].std()
caliper_val = CALIPER * logit_sd
print(f"Logit PS SD = {logit_sd:.4f}  ->  caliper = {caliper_val:.4f}")

# ---------------------------------------------------------------------------
# 3. OPTIMAL 1:1 matching on LOGIT distance, with LOGIT caliper
# ---------------------------------------------------------------------------
gp   = df[df[TREAT_COL] == 1].reset_index(drop=True)
comp = df[df[TREAT_COL] == 0].reset_index(drop=True)
n_gp_eligible = len(gp)

# logit-distance cost matrix
cost = np.abs(gp["logit_ps"].values[:, None] - comp["logit_ps"].values[None, :])

# apply caliper: forbid pairs beyond the caliper
forbidden = cost > caliper_val
cost_gated = cost.copy()
cost_gated[forbidden] = np.inf

# drop GP patients with NO eligible comparator inside the caliper
has_match = ~np.all(np.isinf(cost_gated), axis=1)
dropped_by_caliper = int((~has_match).sum())
gp_rows = np.where(has_match)[0]
cost_gated = cost_gated[gp_rows]

# Hungarian needs finite costs; use a large penalty for forbidden pairs, then
# discard any pair that ended up on a penalty cell (i.e. outside caliper)
BIG = 1e6
solve = np.where(np.isinf(cost_gated), BIG, cost_gated)
r, c = linear_sum_assignment(solve)
keep = [(gp_rows[i], c[k]) for k, i in enumerate(r) if cost_gated[i, c[k]] < BIG]

gp_sel  = [k[0] for k in keep]
comp_sel = [k[1] for k in keep]
gp_m   = gp.loc[gp_sel].reset_index(drop=True)
comp_m = comp.loc[comp_sel].reset_index(drop=True)

pairs = pd.DataFrame({
    "gp_id":   gp_m[ID_COL].values,
    "comp_id": comp_m[ID_COL].values,
    "gp_logit_ps":   gp_m["logit_ps"].values,
    "comp_logit_ps": comp_m["logit_ps"].values,
})
pairs["dist_logit"] = np.abs(pairs["gp_logit_ps"] - pairs["comp_logit_ps"])
pairs.to_csv("optimal_logit_caliper_pairs.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Balance
# ---------------------------------------------------------------------------
rows = []
for cvar in COVARIATES:
    s = abs(smd(gp_m[cvar], comp_m[cvar]))
    rows.append({"covariate": cvar, "smd": s, "balanced": s < 0.1})
bal = pd.DataFrame(rows)
bal.to_csv("optimal_logit_caliper_balance.csv", index=False)
n_bal = int(bal["balanced"].sum())

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
with open("optimal_logit_caliper_report.txt", "w") as f:
    f.write("OPTIMAL MATCHING + LOGIT CALIPER (0.2 SD) — final config\n")
    f.write("unpenalized logistic | logit distance | logit caliper | SMD<0.1\n")
    f.write("="*66 + "\n\n")
    f.write(f"GP eligible (complete-case): {n_gp_eligible}\n")
    f.write(f"Dropped by caliper (no comparator within {caliper_val:.4f}): "
            f"{dropped_by_caliper}\n")
    f.write(f"Final matched pairs: {len(pairs)}\n\n")

    f.write("Within-pair logit distance:\n")
    f.write(f"  median {pairs['dist_logit'].median():.4f} | "
            f"mean {pairs['dist_logit'].mean():.4f} | "
            f"max {pairs['dist_logit'].max():.4f}\n\n")

    f.write(f"Balance: {n_bal}/{len(COVARIATES)} covariates SMD < 0.1\n\n")
    imb = bal[~bal["balanced"]].sort_values("smd", ascending=False)
    f.write("Imbalanced covariates (SMD >= 0.1):\n")
    if len(imb) == 0:
        f.write("  none — all balanced\n")
    else:
        for _, r2 in imb.iterrows():
            f.write(f"  {r2['covariate']:<28} {r2['smd']:.4f}\n")
    f.write("\nPriority covariates:\n")
    for cvar in ["preoperative_bmi", "any_insulin", "rapid_insulin", "long_insulin", "t2dm"]:
        v = bal.loc[bal.covariate == cvar, "smd"].values[0]
        f.write(f"  {cvar:<22} {v:.4f}{'' if v < 0.1 else '  OVER'}\n")

    f.write("\nFull balance table:\n")
    for _, r2 in bal.iterrows():
        f.write(f"  {r2['covariate']:<28} {r2['smd']:.4f}"
                f"{'' if r2['balanced'] else '  OVER'}\n")

print("\n" + open("optimal_logit_caliper_report.txt").read())
