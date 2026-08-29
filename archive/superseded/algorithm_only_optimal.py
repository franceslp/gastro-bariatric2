"""
algorithm_only_optimal.py

Isolates the MATCHING-ALGORITHM effect from the PS-MODEL/SCALE effect.

Runs optimal (Hungarian) matching using the SAME propensity score as the
greedy primary analysis:
    - L2-penalized LogisticRegression(max_iter=1000, random_state=42)
    - StandardScaler on covariates
    - distance on logit(PS)
...changing ONLY greedy -> optimal (still no caliper, to match the paper run).

This lets us decompose the 201/222 reassignment seen in the paper run into:
    (a) how much is caused by the algorithm alone (this script), vs.
    (b) how much is added by also switching the PS model + distance scale
        (the difference between this script and the paper run).

Compares reassignment against greedy TWO ways:
    - over all optimal pairs
    - restricted to the GP patients greedy also matched (apples-to-apples,
      since no-caliper optimal matches more GP than caliper greedy)

Outputs:
    algo_only_matched_pairs.csv
    algo_only_decomposition.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

INPUT_MATRIX = "psm_full_covariate_matrix.csv"
GREEDY_PAIRS = "psm_matched_pairs_new.csv"
PAPER_PAIRS  = "paper_optimal_matched_pairs.csv"   # for the 3-way comparison
ID_COL    = "patient_id"
TREAT_COL = "group_encoded"

# Same 31 covariates as PSM_COVARIATES
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
# 1. Load + complete-case (identical to greedy pipeline)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_MATRIX)
before = len(df)
df = df.dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
print(f"Complete cases: {len(df)} (dropped {before - len(df)})")
y = df[TREAT_COL].astype(int).values
print(f"GP: {(y==1).sum()}  Comparator: {(y==0).sum()}")

# ---------------------------------------------------------------------------
# 2. Propensity score — IDENTICAL to greedy: L2 logistic + StandardScaler,
#    distance on logit(PS). (This is the whole point — same PS as greedy.)
# ---------------------------------------------------------------------------
X = df[COVARIATES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression(max_iter=1000, random_state=42)   # L2 default, matches greedy
lr.fit(X_scaled, y)
ps = np.clip(lr.predict_proba(X_scaled)[:, 1], 1e-6, 1 - 1e-6)
df["propensity_score"] = ps
df["logit_ps"] = np.log(ps / (1 - ps))

# ---------------------------------------------------------------------------
# 3. Optimal (Hungarian) on LOGIT-PS distance, no caliper
#    Only difference from greedy is greedy -> optimal.
# ---------------------------------------------------------------------------
gp   = df[df[TREAT_COL] == 1].reset_index(drop=True)
comp = df[df[TREAT_COL] == 0].reset_index(drop=True)
cost = np.abs(gp["logit_ps"].values[:, None] - comp["logit_ps"].values[None, :])
row_idx, col_idx = linear_sum_assignment(cost)

pairs = pd.DataFrame({
    "gp_id":   gp.loc[row_idx, ID_COL].values,
    "comp_id": comp.loc[col_idx, ID_COL].values,
})
pairs["dist_logit"] = np.abs(
    gp.loc[row_idx, "logit_ps"].values - comp.loc[col_idx, "logit_ps"].values)
pairs.to_csv("algo_only_matched_pairs.csv", index=False)
print(f"Algorithm-only optimal pairs: {len(pairs)}")

# balance
gp_m, comp_m = gp.loc[row_idx].reset_index(drop=True), comp.loc[col_idx].reset_index(drop=True)
n_bal = sum(abs(smd(gp_m[c], comp_m[c])) < 0.1 for c in COVARIATES)

# ---------------------------------------------------------------------------
# 4. Decomposition report
# ---------------------------------------------------------------------------
greedy = pd.read_csv(GREEDY_PAIRS)
g_map = dict(zip(greedy["gp_patient_id"].astype(str),
                 greedy["comp_patient_id"].astype(str)))
greedy_gp = set(g_map.keys())

def reassignment(opt_pairs, label):
    """Compare an optimal run's pairs against greedy's comparator assignment."""
    o_map = dict(zip(opt_pairs["gp_id"].astype(str), opt_pairs["comp_id"].astype(str)))
    shared = greedy_gp & set(o_map.keys())          # GP matched by both
    same   = sum(1 for g in shared if g_map[g] == o_map[g])
    diff   = len(shared) - same
    lines = [
        f"[{label}]",
        f"  optimal pairs total: {len(opt_pairs)}",
        f"  GP matched by both greedy and this: {len(shared)}",
        f"  same comparator:      {same}",
        f"  DIFFERENT comparator: {diff}  ({100*diff/len(shared):.1f}% of shared)",
    ]
    return "\n".join(lines), diff, len(shared)

with open("algo_only_decomposition.txt", "w") as f:
    f.write("REASSIGNMENT DECOMPOSITION: algorithm effect vs PS-model/scale effect\n")
    f.write("="*70 + "\n\n")
    f.write("All runs compared against the greedy primary match.\n")
    f.write("Greedy PS = L2 logistic + scaled + logit distance + caliper.\n\n")

    f.write(f"Algorithm-only balance: {n_bal}/31 covariates SMD<0.1\n\n")

    # (a) algorithm-only reassignment
    txt_a, diff_a, shared_a = reassignment(pairs, "ALGORITHM-ONLY  (same PS as greedy, optimal, no caliper)")
    f.write(txt_a + "\n\n")

    # (b) paper run reassignment (all 3 changes)
    if os.path.exists(PAPER_PAIRS):
        paper = pd.read_csv(PAPER_PAIRS)
        txt_b, diff_b, shared_b = reassignment(paper, "PAPER RUN  (unpenalized + PS-scale + optimal, no caliper)")
        f.write(txt_b + "\n\n")

        f.write("-"*70 + "\n")
        f.write("DECOMPOSITION\n")
        f.write(f"  Reassignment from ALGORITHM alone:        {diff_a}/{shared_a} "
                f"({100*diff_a/shared_a:.1f}%)\n")
        f.write(f"  Reassignment from ALL THREE changes:      {diff_b}/{shared_b} "
                f"({100*diff_b/shared_b:.1f}%)\n")
        extra = diff_b - diff_a
        f.write(f"  Added by PS-model + distance-scale change: ~{extra} more GP\n\n")
        if diff_a > 0:
            share = 100*diff_a/diff_b if diff_b else 0
            f.write(f"  => The matching algorithm alone accounts for ~{share:.0f}% "
                    f"of the total reassignment.\n")
            f.write(f"     The rest comes from the propensity score itself moving\n")
            f.write(f"     (unpenalized vs L2, PS-scale vs logit).\n")

    # also: how much do algo-only and paper agree with EACH OTHER?
    if os.path.exists(PAPER_PAIRS):
        a_map = dict(zip(pairs["gp_id"].astype(str), pairs["comp_id"].astype(str)))
        p_map = dict(zip(paper["gp_id"].astype(str), paper["comp_id"].astype(str)))
        both = set(a_map) & set(p_map)
        agree = sum(1 for g in both if a_map[g] == p_map[g])
        f.write(f"\n  (cross-check) algorithm-only vs paper run agree on "
                f"{agree}/{len(both)} shared GP comparators\n")

print("\nWrote algo_only_decomposition.txt")

# echo to console
print("\n" + open("algo_only_decomposition.txt").read())
