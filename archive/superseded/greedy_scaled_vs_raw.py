"""
greedy_scaled_vs_raw.py

Replicates your ACTUAL greedy pipeline (assemble_and_run_psm.py) exactly:
  - L2-penalized LogisticRegression(max_iter=1000, random_state=42)
  - distance on logit(PS)
  - 1:1 greedy nearest-neighbor matching, PS-sorted
  - caliper = 0.2 SD of logit PS
  - complete-case, no imputation

...and runs it TWICE: once with StandardScaler (your current setup) and once
with RAW inputs (no scaling). Reports balance on all 31 covariates for each,
so you can see whether dropping the scaler balances BMI and insulin WITH the
caliper on (the grid earlier was no-caliper optimal; this is your real method).

Reads only existing files. Installs nothing.
Output: greedy_scaled_vs_raw_report.txt
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

INPUT_MATRIX = "psm_full_covariate_matrix.csv"
ID_COL, TREAT_COL = "patient_id", "group_encoded"
CALIPER = 0.2   # SD of logit PS, per Austin 2011 — matches your pipeline

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
PRIORITY = ["preoperative_bmi", "any_insulin", "rapid_insulin", "long_insulin"]

def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled

df = pd.read_csv(INPUT_MATRIX).dropna(subset=COVARIATES + [TREAT_COL]).reset_index(drop=True)
y = df[TREAT_COL].astype(int).values
X_raw = df[COVARIATES].values.astype(float)

def greedy_match(scaling):
    """Run your exact greedy pipeline with either scaled or raw inputs."""
    X = StandardScaler().fit_transform(X_raw) if scaling == "scaled" else X_raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=1000, random_state=42)   # L2, matches pipeline
        lr.fit(X, y)
    ps = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    d = df.copy()
    d["ps"] = ps
    d["logit_ps"] = np.log(ps / (1 - ps))

    logit_sd = d["logit_ps"].std()
    caliper_val = CALIPER * logit_sd

    gp   = d[d[TREAT_COL] == 1].copy().reset_index(drop=True)
    comp = d[d[TREAT_COL] == 0].copy().reset_index(drop=True)

    tree = KDTree(comp[["logit_ps"]].values)
    used = set()
    pairs = []
    # PS-sorted greedy, exactly like assemble_and_run_psm.py
    gp_sorted = gp.sort_values("ps").reset_index(drop=True)
    for _, row in gp_sorted.iterrows():
        gl = row["logit_ps"]
        dists, idxs = tree.query([[gl]], k=min(10, len(comp)))
        for dist, idx in zip(dists[0], idxs[0]):
            if dist <= caliper_val and idx not in used:
                pairs.append((row[ID_COL], comp.loc[idx, ID_COL], idx, row.name))
                used.add(idx)
                break
    gp_idx  = [gp_sorted.index[gp_sorted[ID_COL] == p[0]][0] for p in pairs]
    gp_m    = gp_sorted.loc[gp_idx].reset_index(drop=True)
    comp_m  = comp.loc[[p[2] for p in pairs]].reset_index(drop=True)

    smds = {c: abs(smd(gp_m[c], comp_m[c])) for c in COVARIATES}
    return len(pairs), caliper_val, logit_sd, smds

n_s, cal_s, sd_s, smd_s = greedy_match("scaled")
n_r, cal_r, sd_r, smd_r = greedy_match("raw")

with open("greedy_scaled_vs_raw_report.txt", "w") as f:
    f.write("GREEDY PIPELINE — scaled inputs (current) vs raw inputs\n")
    f.write("Your exact method: L2, logit distance, 0.2 SD caliper, greedy 1:1.\n")
    f.write("="*66 + "\n\n")

    f.write(f"{'':<22}{'SCALED (current)':>20}{'RAW':>16}\n")
    f.write(f"{'pairs matched':<22}{n_s:>20}{n_r:>16}\n")
    f.write(f"{'caliper value':<22}{cal_s:>20.4f}{cal_r:>16.4f}\n")
    nbal_s = sum(v < 0.1 for v in smd_s.values())
    nbal_r = sum(v < 0.1 for v in smd_r.values())
    f.write(f"{'covariates <0.1':<22}{str(nbal_s)+'/31':>20}{str(nbal_r)+'/31':>16}\n\n")

    f.write("PRIORITY COVARIATES (your outcome-critical ones):\n")
    f.write(f"{'covariate':<22}{'SCALED':>20}{'RAW':>16}\n")
    for c in PRIORITY:
        flag_s = "" if smd_s[c] < 0.1 else "  OVER"
        flag_r = "" if smd_r[c] < 0.1 else "  OVER"
        f.write(f"{c:<22}{smd_s[c]:>14.3f}{flag_s:<6}{smd_r[c]:>10.3f}{flag_r}\n")
    f.write("\n")

    f.write("ALL imbalanced covariates (SMD >= 0.1):\n")
    imb_s = {c: smd_s[c] for c in COVARIATES if smd_s[c] >= 0.1}
    imb_r = {c: smd_r[c] for c in COVARIATES if smd_r[c] >= 0.1}
    f.write(f"  SCALED: {', '.join(f'{k}={v:.3f}' for k,v in imb_s.items()) or 'none'}\n")
    f.write(f"  RAW:    {', '.join(f'{k}={v:.3f}' for k,v in imb_r.items()) or 'none'}\n\n")

    f.write("FULL per-covariate SMD:\n")
    f.write(f"{'covariate':<22}{'SCALED':>12}{'RAW':>12}{'change':>12}\n")
    for c in COVARIATES:
        f.write(f"{c:<22}{smd_s[c]:>12.3f}{smd_r[c]:>12.3f}{smd_r[c]-smd_s[c]:>12.3f}\n")

print(open("greedy_scaled_vs_raw_report.txt").read())
