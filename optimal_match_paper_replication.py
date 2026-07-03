"""
optimal_match_paper_replication.py

Replicates the OPTIMAL matching method of Kendall et al. (Am Surg 2025) as
closely as possible in Python (scipy + sklearn only — no new installs).

WHAT THE PAPER DOES (MatchIt method="optimal"):
  - Propensity score from UNPENALIZED logistic regression (R glm)
  - 1:1 optimal matching that minimizes TOTAL within-pair propensity-score
    distance (via optmatch network-flow solver)
  - No caliper in the worked example
  - Distance on the PROPENSITY SCORE scale (not logit)

HOW THIS SCRIPT MATCHES THAT:
  - Unpenalized logistic regression: LogisticRegression(C=np.inf)  == R glm
  - 1:1 optimal assignment via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment). For exact 1:1 pair matching the
    Hungarian algorithm returns the SAME optimum as optmatch — same
    optimization problem, different solver. This is an EQUIVALENT
    IMPLEMENTATION of 1:1 optimal matching, not the identical software.
  - No caliper
  - Distance on the propensity-score scale

NOTE ON COMPARISON:
  This intentionally DIFFERS from the greedy pipeline in three ways at once
  (unpenalized vs L2 PS; PS-scale vs logit-scale distance; optimal vs greedy).
  That bundle IS "the paper's method vs. mine." Do not read per-covariate
  differences as attributable to any single one of those three changes.

OUTPUTS:
  paper_optimal_matched_pairs.csv     gp_id, comp_id, gp_ps, comp_ps, dist
  paper_optimal_smd_after.csv         SMD per covariate after optimal matching
  paper_optimal_vs_greedy_summary.txt balance + pair overlap + distance stats
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_MATRIX = "psm_full_covariate_matrix.csv"   # all eligible patients, pre-match, with-BMI
GREEDY_PAIRS = "psm_matched_pairs_new.csv"       # existing greedy pairs (for overlap)
ID_COL       = "patient_id"
TREAT_COL    = "group_encoded"                    # 1 = GP (treated), 0 = comparator
EPS          = 1e-6                               # clip PS away from 0/1 (see below)

# 31 covariates — EXACTLY matches PSM_COVARIATES in assemble_and_run_psm.py
# (surgery_year is in the CSV but NOT in the PS model — excluded here too)
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
    """Standardized mean difference (pooled SD; std convention for binaries)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled

# ---------------------------------------------------------------------------
# 1. Load + complete-case (no imputation)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_MATRIX)
print(f"Loaded {len(df)} patients, {df.shape[1]} columns")

missing = [c for c in COVARIATES + [TREAT_COL, ID_COL] if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

before = len(df)
df = df.dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
print(f"Complete cases: {len(df)} (dropped {before - len(df)})")

y = df[TREAT_COL].astype(int).values
print(f"Treated (GP): {(y==1).sum()}  Comparator: {(y==0).sum()}")

# ---------------------------------------------------------------------------
# 2. Propensity score — UNPENALIZED logistic regression (== R glm, per paper)
#    NOTE: paper's MatchIt default fits glm on RAW covariates (no scaling).
#    We follow that here (no StandardScaler) to stay faithful to the paper.
# ---------------------------------------------------------------------------
X = df[COVARIATES].values.astype(float)
# Unpenalized logistic == R glm. sklearn maps this internally; the exact
# keyword differs slightly by version. We set penalty=None and silence the
# version-specific deprecation notice (behavior is correct either way).
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = LogisticRegression(penalty=None, max_iter=5000, random_state=42)
    lr.fit(X, y)
if lr.n_iter_[0] >= 5000:
    print("WARNING: logistic regression may not have converged "
          "(possible separation). Inspect coefficients.")

ps = lr.predict_proba(X)[:, 1]

# --- reviewer fix #1: clip PS away from exactly 0/1 to avoid inf downstream.
# Optimal matching here uses PS-scale distance so logit isn't strictly needed,
# but clip anyway for safety and for the logit reported in diagnostics.
ps = np.clip(ps, EPS, 1 - EPS)
df["propensity_score"] = ps
df["logit_ps"] = np.log(ps / (1 - ps))   # for diagnostics only

# ---------------------------------------------------------------------------
# 3. OPTIMAL 1:1 matching (Hungarian) on PROPENSITY-SCORE distance, NO caliper
# ---------------------------------------------------------------------------
gp   = df[df[TREAT_COL] == 1].reset_index(drop=True)
comp = df[df[TREAT_COL] == 0].reset_index(drop=True)
print(f"\n--- OPTIMAL 1:1 matching (Hungarian, PS distance, no caliper) ---")
print(f"GP: {len(gp)}  Comparator pool: {len(comp)}")

# Cost = |ps_gp - ps_comp|, shape (n_gp, n_comp). Memory: n_gp*n_comp floats.
cost = np.abs(gp["propensity_score"].values[:, None]
              - comp["propensity_score"].values[None, :])
print(f"Cost matrix shape: {cost.shape} "
      f"(~{cost.nbytes/1e6:.1f} MB)")

row_idx, col_idx = linear_sum_assignment(cost)   # optimal, minimizes total distance

pairs = pd.DataFrame({
    "gp_id":   gp.loc[row_idx, ID_COL].values,
    "comp_id": comp.loc[col_idx, ID_COL].values,
    "gp_ps":   gp.loc[row_idx, "propensity_score"].values,
    "comp_ps": comp.loc[col_idx, "propensity_score"].values,
})
pairs["dist"] = np.abs(pairs["gp_ps"] - pairs["comp_ps"])
pairs.to_csv("paper_optimal_matched_pairs.csv", index=False)

# --- reviewer fixes #3 + #4: report total (the optimized quantity) + median/IQR
total_d  = pairs["dist"].sum()
mean_d   = pairs["dist"].mean()
median_d = pairs["dist"].median()
q1, q3   = pairs["dist"].quantile([0.25, 0.75])
max_d    = pairs["dist"].max()
print(f"Optimal matched pairs: {len(pairs)}")
print(f"TOTAL within-pair PS distance (optimized quantity): {total_d:.4f}")
print(f"Mean {mean_d:.4f} | Median {median_d:.4f} "
      f"| IQR [{q1:.4f}, {q3:.4f}] | Max {max_d:.4f}")

# ---------------------------------------------------------------------------
# 4. Balance after optimal matching
# ---------------------------------------------------------------------------
gp_m   = gp.loc[row_idx].reset_index(drop=True)
comp_m = comp.loc[col_idx].reset_index(drop=True)

rows = []
for c in COVARIATES:
    rows.append({
        "covariate": c,
        "gp_mean":   gp_m[c].mean(),
        "comp_mean": comp_m[c].mean(),
        "smd_post":  abs(smd(gp_m[c], comp_m[c])),
    })
bal = pd.DataFrame(rows)
bal["balanced_post"] = bal["smd_post"] < 0.1
bal.to_csv("paper_optimal_smd_after.csv", index=False)
n_bal = int(bal["balanced_post"].sum())
print(f"\nBalance after optimal match: {n_bal}/{len(COVARIATES)} covariates SMD<0.1")

# ---------------------------------------------------------------------------
# 5. Compare vs greedy pairs
# ---------------------------------------------------------------------------
with open("paper_optimal_vs_greedy_summary.txt", "w") as f:
    f.write("PAPER-REPLICATION OPTIMAL (unpenalized glm, PS distance, Hungarian, "
            "no caliper)\n")
    f.write("   vs GREEDY (L2 logistic, logit distance, nearest-neighbor + 0.2 caliper)\n")
    f.write("="*74 + "\n\n")
    f.write("NOTE: differs from greedy on 3 axes (PS penalty, distance scale, "
            "algorithm).\nThis is 'paper method vs mine', not an isolated "
            "algorithm test.\n\n")

    f.write(f"Optimal matched pairs: {len(pairs)}\n")
    f.write(f"Optimal balance: {n_bal}/{len(COVARIATES)} covariates SMD<0.1\n\n")
    f.write("Within-pair PS distance:\n")
    f.write(f"  total  {total_d:.4f}  (this is what optimal matching minimizes)\n")
    f.write(f"  mean   {mean_d:.4f}\n")
    f.write(f"  median {median_d:.4f}   IQR [{q1:.4f}, {q3:.4f}]\n")
    f.write(f"  max    {max_d:.4f}\n\n")

    f.write("Covariates still imbalanced (SMD>=0.1) after OPTIMAL match:\n")
    imb = bal[~bal["balanced_post"]]
    if len(imb) == 0:
        f.write("  (none — all 31 balanced)\n")
    else:
        for _, r in imb.iterrows():
            f.write(f"  {r['covariate']:<32} SMD={r['smd_post']:.4f}\n")
    f.write("\n")

    if os.path.exists(GREEDY_PAIRS):
        g = pd.read_csv(GREEDY_PAIRS)
        gp_c, comp_c = "gp_patient_id", "comp_patient_id"
        f.write(f"Greedy pairs file columns: {list(g.columns)}\n")
        if gp_c in g.columns and comp_c in g.columns:
            g_set = set(zip(g[gp_c].astype(str), g[comp_c].astype(str)))
            o_set = set(zip(pairs["gp_id"].astype(str), pairs["comp_id"].astype(str)))
            exact = len(g_set & o_set)
            gp_shared = len(set(g[gp_c].astype(str)) & set(pairs["gp_id"].astype(str)))
            f.write(f"\n--- PAIR OVERLAP ---\n")
            f.write(f"Greedy pairs: {len(g)}  Optimal pairs: {len(pairs)}\n")
            f.write(f"Identical (gp+comp) pairs in both: {exact} "
                    f"({100*exact/len(pairs):.1f}% of optimal)\n")
            f.write(f"GP patients matched in both methods: {gp_shared}\n")
            f.write(f"GP matched to a DIFFERENT comparator under optimal: "
                    f"{gp_shared - exact}\n")
        else:
            f.write("Could not find expected id columns in greedy pairs file.\n")
    else:
        f.write(f"Greedy pairs file '{GREEDY_PAIRS}' not found — overlap skipped.\n")

print("\nDone. Outputs:")
print("  paper_optimal_matched_pairs.csv")
print("  paper_optimal_smd_after.csv")
print("  paper_optimal_vs_greedy_summary.txt")
