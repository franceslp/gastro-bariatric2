"""
match_diagnostics.py

Two diagnostics for the PSM method comparison:

(1) BAD-PAIR COUNT — how many optimal-matched pairs would have VIOLATED the
    greedy caliper (0.2 SD of logit PS)? i.e. matches greedy would have refused.
    Measured on YOUR logit/L2 scale so it's a fair comparison against your
    actual caliper.

(2) OVERFITTING / SEPARATION CHECK — compares the L2-penalized model (yours)
    against the unpenalized model (paper's) for warning signs:
      - propensity scores pinned near 0 or 1 (separation signature)
      - extreme regression coefficients (fluke-chasing signature)
      - convergence (non-convergence often signals separation)
      - how score mass distributes in the tails

Reads only files already present; installs nothing.

Outputs:
    caliper_violation_report.txt
    model_separation_report.txt
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
CALIPER = 0.2   # SD of logit PS — your greedy caliper multiplier

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

# ---------------------------------------------------------------------------
# Load + complete-case
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_MATRIX).dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
y = df[TREAT_COL].astype(int).values
X = df[COVARIATES].values

def fit_ps(penalized):
    """Return (propensity, logit_ps, coefs, n_iter, converged) for a model."""
    Xs = StandardScaler().fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if penalized:
            lr = LogisticRegression(max_iter=1000, random_state=42)          # L2 (yours)
        else:
            lr = LogisticRegression(penalty=None, max_iter=5000, random_state=42)  # unpenalized (paper)
        lr.fit(Xs, y)
    ps = np.clip(lr.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(ps / (1 - ps))
    converged = lr.n_iter_[0] < (1000 if penalized else 5000)
    return ps, logit, lr.coef_[0], int(lr.n_iter_[0]), converged

# both models on the SAME complete-case population
ps_L2,  logit_L2,  coef_L2,  it_L2,  conv_L2  = fit_ps(penalized=True)
ps_UNP, logit_UNP, coef_UNP, it_UNP, conv_UNP = fit_ps(penalized=False)

df["ps_L2"], df["logit_L2"] = ps_L2, logit_L2

# caliper value on YOUR logit/L2 scale (this is what greedy used)
logit_sd_L2 = np.std(logit_L2, ddof=1)
caliper_val = CALIPER * logit_sd_L2

# ===========================================================================
# DIAGNOSTIC 1 — bad-pair count (optimal pairs that violate greedy caliper)
# ===========================================================================
# Build the algorithm-only optimal match (same L2/logit scores, optimal, no caliper)
gp   = df[df[TREAT_COL] == 1].reset_index(drop=True)
comp = df[df[TREAT_COL] == 0].reset_index(drop=True)
cost = np.abs(gp["logit_L2"].values[:, None] - comp["logit_L2"].values[None, :])
r, c = linear_sum_assignment(cost)
opt = pd.DataFrame({
    "gp_id":   gp.loc[r, ID_COL].values,
    "comp_id": comp.loc[c, ID_COL].values,
    "dist_logit_L2": np.abs(gp.loc[r, "logit_L2"].values - comp.loc[c, "logit_L2"].values),
})

viol = opt[opt["dist_logit_L2"] > caliper_val]

with open("caliper_violation_report.txt", "w") as f:
    f.write("DIAGNOSTIC 1 — BAD-PAIR COUNT (optimal vs greedy caliper)\n")
    f.write("="*64 + "\n\n")
    f.write(f"Greedy caliper = {CALIPER} x SD(logit PS)\n")
    f.write(f"SD(logit L2) = {logit_sd_L2:.4f}  ->  caliper distance = {caliper_val:.4f}\n\n")
    f.write("Optimal match (same L2/logit scores as greedy, optimal, NO caliper):\n")
    f.write(f"  total optimal pairs: {len(opt)}\n")
    f.write(f"  pairs EXCEEDING the greedy caliper (i.e. greedy would refuse): "
            f"{len(viol)}  ({100*len(viol)/len(opt):.1f}%)\n\n")
    f.write("Distance summary of optimal pairs (logit L2 scale):\n")
    f.write(f"  min {opt['dist_logit_L2'].min():.4f} | median {opt['dist_logit_L2'].median():.4f} "
            f"| mean {opt['dist_logit_L2'].mean():.4f} | max {opt['dist_logit_L2'].max():.4f}\n\n")
    if len(viol):
        f.write(f"The {len(viol)} violating pairs (distance vs caliper {caliper_val:.4f}):\n")
        for _, row in viol.sort_values("dist_logit_L2", ascending=False).iterrows():
            f.write(f"  gp={row['gp_id']}  comp={row['comp_id']}  "
                    f"dist={row['dist_logit_L2']:.4f}  "
                    f"({row['dist_logit_L2']/caliper_val:.1f}x caliper)\n")
    else:
        f.write("No optimal pairs exceed the caliper — optimal happened to stay within bounds.\n")
    # context: greedy dropped how many?
    g = pd.read_csv(GREEDY_PAIRS)
    f.write(f"\nFor context: greedy kept {len(g)} pairs (dropped "
            f"{gp.shape[0]-len(g)} GP via caliper); optimal kept {len(opt)}.\n")

# ===========================================================================
# DIAGNOSTIC 2 — overfitting / separation check (L2 vs unpenalized)
# ===========================================================================
def tail_counts(ps):
    return {
        "<0.001":  int((ps < 0.001).sum()),
        "<0.01":   int((ps < 0.01).sum()),
        ">0.99":   int((ps > 0.99).sum()),
        ">0.999":  int((ps > 0.999).sum()),
    }

with open("model_separation_report.txt", "w") as f:
    f.write("DIAGNOSTIC 2 — OVERFITTING / SEPARATION CHECK\n")
    f.write("="*64 + "\n\n")
    f.write(f"Population: {len(df)} patients, {len(COVARIATES)} covariates, "
            f"{int(y.sum())} GP / {int((1-y).sum())} comparator\n")
    f.write("Rule of thumb: with many covariates and few events, unpenalized\n")
    f.write("logistic can overfit or separate. Signs = extreme scores, huge\n")
    f.write("coefficients, non-convergence. L2 should show fewer of these.\n\n")

    f.write("--- Convergence ---\n")
    f.write(f"  L2 (yours):        iterations={it_L2}, converged={conv_L2}\n")
    f.write(f"  Unpenalized (paper): iterations={it_UNP}, converged={conv_UNP}\n")
    if not conv_UNP:
        f.write("  ** Unpenalized did NOT converge — classic separation warning. **\n")
    f.write("\n")

    f.write("--- Extreme propensity scores (separation signature) ---\n")
    tL2, tU = tail_counts(ps_L2), tail_counts(ps_UNP)
    f.write(f"  {'threshold':<10}{'L2 (yours)':>14}{'Unpenalized':>14}\n")
    for k in tL2:
        f.write(f"  {k:<10}{tL2[k]:>14}{tU[k]:>14}\n")
    f.write("  (more scores jammed at the extremes under unpenalized = separation)\n\n")

    f.write("--- Coefficient magnitude (fluke-chasing signature) ---\n")
    f.write(f"  {'stat':<22}{'L2 (yours)':>14}{'Unpenalized':>14}\n")
    f.write(f"  {'max |coef|':<22}{np.abs(coef_L2).max():>14.3f}{np.abs(coef_UNP).max():>14.3f}\n")
    f.write(f"  {'mean |coef|':<22}{np.abs(coef_L2).mean():>14.3f}{np.abs(coef_UNP).mean():>14.3f}\n")
    f.write(f"  {'# |coef|>5':<22}{int((np.abs(coef_L2)>5).sum()):>14}{int((np.abs(coef_UNP)>5).sum()):>14}\n")
    f.write(f"  {'# |coef|>10':<22}{int((np.abs(coef_L2)>10).sum()):>14}{int((np.abs(coef_UNP)>10).sum()):>14}\n\n")

    # biggest coefficient gaps between the two models
    gap = pd.DataFrame({
        "covariate": COVARIATES,
        "coef_L2":   coef_L2,
        "coef_unpen": coef_UNP,
        "abs_diff":  np.abs(coef_L2 - coef_UNP),
    }).sort_values("abs_diff", ascending=False)
    f.write("--- Covariates whose weight changed most between models (top 8) ---\n")
    f.write(f"  {'covariate':<28}{'L2':>9}{'unpen':>9}{'|diff|':>9}\n")
    for _, row in gap.head(8).iterrows():
        f.write(f"  {row['covariate']:<28}{row['coef_L2']:>9.2f}"
                f"{row['coef_unpen']:>9.2f}{row['abs_diff']:>9.2f}\n")

print(open("caliper_violation_report.txt").read())
print("\n" + "="*64 + "\n")
print(open("model_separation_report.txt").read())
