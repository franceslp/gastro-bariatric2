"""
run_PSM.py

Propensity Score Matching (PSM) - randomized greedy 1:1 nearest-neighbor
matching with caliper.

Method:
  1. Logistic regression on covariates to estimate propensity scores
  2. Randomized greedy nearest-neighbor matching - matching order randomized
     per Austin 2011 to avoid systematic bias from sequential processing.
     Caliper applied before selecting nearest match among eligible controls.
  3. Caliper = 0.2 * SD of logit(PS) per Austin 2011 / Sadda et al.
  4. SMD reported before and after matching for balance assessment

Note: this is randomized greedy matching, not optimal bipartite matching.
With a ~62:1 control-to-case ratio, greedy vs. optimal produces negligible
differences in practice.

PSM covariates:
  - age_at_surgery_approx (continuous)
  - sex (binary encoded: F=1)
  - surgery_type (sleeve=1, bypass=0)
  - has_diabetes (binary)
  - surgery_year (one-hot encoded - categorical, not linear)
  - BMI_at_or_before_surgery (continuous)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GP_CSV = "funnel_6_final_cohort.csv"
GP_BMI_CSV = "gastroparesis_cohort_BMI_at_or_before_surgery.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM_with_BMI.csv"
OUTPUT_MATCHED = "psm_matched_pairs.csv"
OUTPUT_ALL_SCORES = "psm_all_candidates_with_scores.csv"

CALIPER_MULTIPLIER = 0.2
RANDOM_STATE = 42

SLEEVE_CODES = {"43775"}
BYPASS_CODES = {"43644", "43645", "43846", "43847"}

print(">>> SCRIPT VERSION: run_PSM_v3 <<<")

# --- Load and prepare study group ---
gp = pd.read_csv(GP_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "bariatric_date", "sex",
                            "bariatric_cpt_codes_seen", "diabetes_type_label",
                            "age_at_surgery_approx", "year_of_birth"])
gp_bmi = pd.read_csv(GP_BMI_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "BMI_at_or_before_surgery"])
gp = gp.merge(gp_bmi, on="patient_id", how="left")

def get_surgery_type(codes_str):
    if pd.isna(codes_str):
        return None
    codes = set(codes_str.split(","))
    if bool(codes & SLEEVE_CODES) and bool(codes & BYPASS_CODES):
        return None
    elif bool(codes & SLEEVE_CODES):
        return "sleeve"
    elif bool(codes & BYPASS_CODES):
        return "bypass"
    return None

gp["surgery_type"] = gp["bariatric_cpt_codes_seen"].apply(get_surgery_type)
gp["surgery_year"] = pd.to_datetime(gp["bariatric_date"], errors="coerce").dt.year
gp["has_diabetes"] = gp["diabetes_type_label"].isin(["Type 1", "Type 2", "Both"])
gp["group"] = 1

print(f"Study group loaded: {len(gp):,} patients")

# --- Load comparator pool ---
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False)
comp["group"] = 0
print(f"Comparator pool loaded: {len(comp):,} patients")

# --- Combine ---
BASE_VARS = ["patient_id", "group", "age_at_surgery_approx", "sex",
             "surgery_type", "has_diabetes", "surgery_year",
             "BMI_at_or_before_surgery"]

combined = pd.concat([gp[BASE_VARS], comp[BASE_VARS]], ignore_index=True)

# Report missingness per covariate before dropping - BMI is the most
# likely driver; report GP vs comparator separately since non-random
# missingness could bias the complete-case matched cohort
print(f"\nBMI missingness before dropping (non-random missingness = selection bias risk):")
print(f"  GP BMI missing:         {combined.loc[combined['group']==1, 'BMI_at_or_before_surgery'].isna().sum():,}/"
      f"{(combined['group']==1).sum():,} "
      f"({100*combined.loc[combined['group']==1, 'BMI_at_or_before_surgery'].isna().mean():.1f}%)")
print(f"  Comparator BMI missing: {combined.loc[combined['group']==0, 'BMI_at_or_before_surgery'].isna().sum():,}/"
      f"{(combined['group']==0).sum():,} "
      f"({100*combined.loc[combined['group']==0, 'BMI_at_or_before_surgery'].isna().mean():.1f}%)")
print("  NOTE: If rates differ substantially, discuss with PI before proceeding.")

print(f"\nMissingness per PSM covariate (% missing):")
print((combined[["age_at_surgery_approx", "sex", "surgery_type",
                  "has_diabetes", "surgery_year",
                  "BMI_at_or_before_surgery"]].isna().mean() * 100
       ).round(1).sort_values(ascending=False))

n_before = len(combined)
combined = combined.dropna(subset=["age_at_surgery_approx", "sex", "surgery_type",
                                    "has_diabetes", "surgery_year",
                                    "BMI_at_or_before_surgery"])
print(f"\nDropped {n_before-len(combined):,} with missing PSM covariates")
print(f"  Study group: {(combined['group']==1).sum():,}")
print(f"  Comparator:  {(combined['group']==0).sum():,}")

# --- Encode covariates ---
assert combined["surgery_type"].isin(["sleeve", "bypass"]).all(), \
    "Unexpected surgery_type values - check Stage 3 output"

combined["sex_encoded"] = (combined["sex"].str.upper() == "F").astype(int)
combined["surgery_type_encoded"] = (combined["surgery_type"] == "sleeve").astype(int)
# fillna(0) before astype ensures no silent NaN bias in SMD mean calculation
combined["has_diabetes_encoded"] = combined["has_diabetes"].fillna(False).astype(int)

# Surgery year as one-hot (categorical not linear)
year_dummies = pd.get_dummies(combined["surgery_year"].astype(int),
                               prefix="yr", drop_first=True)
combined = pd.concat([combined.reset_index(drop=True),
                       year_dummies.reset_index(drop=True)], axis=1)
year_cols = [c for c in combined.columns if c.startswith("yr_")]

NUMERIC_VARS = (["age_at_surgery_approx", "sex_encoded", "surgery_type_encoded",
                  "has_diabetes_encoded", "BMI_at_or_before_surgery"] + year_cols)

X = combined[NUMERIC_VARS].values.astype(float)
y = combined["group"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Fit logistic regression ---
print("\nFitting logistic regression for propensity scores...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_scaled, y)

combined["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]

# Clip both before logit to avoid log(0) or log(inf)
ps_clipped = combined["propensity_score"].clip(1e-6, 1 - 1e-6)
combined["logit_ps"] = np.log(ps_clipped / (1 - ps_clipped))

print(f"Propensity score summary:")
print(f"  Study group:  mean={combined.loc[combined['group']==1,'propensity_score'].mean():.3f}")
print(f"  Comparator:   mean={combined.loc[combined['group']==0,'propensity_score'].mean():.3f}")

# Save audit trail
combined.to_csv(OUTPUT_ALL_SCORES, index=False)
print(f"\nWrote {OUTPUT_ALL_SCORES} (audit trail with all propensity scores)")

# --- SMD function ---
def smd(var, df, group_col="group"):
    g1 = df.loc[df[group_col]==1, var].dropna()
    g0 = df.loc[df[group_col]==0, var].dropna()
    # For explicitly encoded binary variables use absolute difference in
    # proportions - more stable than SD-based when variance is small.
    # Naming convention: all binary PSM vars end in "_encoded"
    if var.endswith("_encoded"):
        return abs(g1.mean() - g0.mean())
    pooled_sd = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
    if pooled_sd == 0:
        return 0.0
    return abs(g1.mean() - g0.mean()) / pooled_sd

SMD_VARS = ["age_at_surgery_approx", "BMI_at_or_before_surgery",
            "sex_encoded", "surgery_type_encoded", "has_diabetes_encoded"]
smd_before_matching = {v: smd(v, combined) for v in SMD_VARS}

print("\nSMD BEFORE matching (target: <0.1 after matching):")
for var in SMD_VARS:
    print(f"  {var}: {smd_before_matching[var]:.3f}")

# --- Overlap / common support diagnostics ---
print("\nOVERLAP DIAGNOSTICS:")
gp_ps = combined.loc[combined["group"]==1, "propensity_score"]
comp_ps = combined.loc[combined["group"]==0, "propensity_score"]
gp_range = (gp_ps.min(), gp_ps.max())
comp_range = (comp_ps.min(), comp_ps.max())
overlap_min = max(gp_range[0], comp_range[0])
overlap_max = min(gp_range[1], comp_range[1])
print(f"  GP PS range:         [{gp_range[0]:.3f}, {gp_range[1]:.3f}]")
print(f"  Comparator PS range: [{comp_range[0]:.3f}, {comp_range[1]:.3f}]")
print(f"  Common support:      [{overlap_min:.3f}, {overlap_max:.3f}]")
gp_outside = ((gp_ps < comp_range[0]) | (gp_ps > comp_range[1])).sum()
print(f"  GP patients outside comparator support: {gp_outside:,}/{len(gp_ps):,} "
      f"({100*gp_outside/len(gp_ps):.1f}%) - these may be hard to match")

# Save PS distributions for external plotting (Love plot, overlap histogram)
ps_summary = pd.DataFrame({
    "patient_id": combined["patient_id"],
    "group": combined["group"],
    "propensity_score": combined["propensity_score"],
    "logit_ps": combined["logit_ps"],
})
ps_summary.to_csv("psm_propensity_scores_by_group.csv", index=False)
print(f"\nWrote psm_propensity_scores_by_group.csv (use for overlap histogram)")

# --- Caliper ---
caliper = CALIPER_MULTIPLIER * combined["logit_ps"].std()
logit_sd = combined["logit_ps"].std()
print(f"\nLogit PS SD: {logit_sd:.4f} (note: inflated by year dummies - expected)")
print(f"Caliper: {caliper:.4f} (0.2 * SD of logit PS)")

gp_pool = combined[combined["group"] == 1].copy().reset_index(drop=True)
comp_pool = combined[combined["group"] == 0].copy().reset_index(drop=True)

# Randomize matching order per Austin 2011
gp_pool = gp_pool.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# --- Greedy nearest-neighbor matching with caliper ---
matched_gp_idx = []
matched_comp_idx = []
unmatched_gp = []
used_comp_indices = set()

for i, gp_row in gp_pool.iterrows():
    available = comp_pool[~comp_pool.index.isin(used_comp_indices)]
    if available.empty:
        unmatched_gp.append(gp_row["patient_id"])
        continue

    diffs = (available["logit_ps"] - gp_row["logit_ps"]).abs()

    # Apply caliper first - filter to eligible matches only
    eligible = available.loc[diffs <= caliper]
    if eligible.empty:
        unmatched_gp.append(gp_row["patient_id"])
        continue

    # Recompute diffs from eligible directly - avoids index arithmetic ambiguity
    best_idx = (eligible["logit_ps"] - gp_row["logit_ps"]).abs().idxmin()
    matched_gp_idx.append(i)
    matched_comp_idx.append(best_idx)
    used_comp_indices.add(best_idx)

print(f"\nMatching results:")
print(f"  Matched pairs:          {len(matched_gp_idx):,}")
print(f"  Unmatched GP patients:  {len(unmatched_gp):,}")
if unmatched_gp:
    print(f"  Unmatched IDs: {unmatched_gp}")

# --- SMD after matching ---
if matched_gp_idx:
    matched_combined = pd.concat([
        gp_pool.loc[matched_gp_idx].assign(group=1),
        comp_pool.loc[matched_comp_idx].assign(group=0)
    ], ignore_index=True)

    SMD_VARS = ["age_at_surgery_approx", "BMI_at_or_before_surgery",
                "sex_encoded", "surgery_type_encoded", "has_diabetes_encoded"]

    # Precompute both to avoid repeated calls and ensure consistency
    smd_before = smd_before_matching  # already computed above
    smd_after = {v: smd(v, matched_combined) for v in SMD_VARS}

    print("\nSMD AFTER matching (target: <0.1):")
    for var in SMD_VARS:
        s = smd_after[var]
        flag = " ✓" if s < 0.1 else " ← REVIEW"
        print(f"  {var}: {s:.3f}{flag}")

    # Save matched pairs
    gp_matched = gp_pool.loc[matched_gp_idx].add_suffix("_gp").reset_index(drop=True)
    comp_matched = comp_pool.loc[matched_comp_idx].add_suffix("_comp").reset_index(drop=True)
    pairs = pd.concat([gp_matched, comp_matched], axis=1)
    pairs["pair_id"] = range(1, len(pairs) + 1)
    pairs.to_csv(OUTPUT_MATCHED, index=False)
    print(f"\nWrote {OUTPUT_MATCHED} ({len(pairs):,} matched pairs)")

    # Effective sample size
    n_gp_original = (combined["group"]==1).sum()
    n_comp_original = (combined["group"]==0).sum()
    print(f"\nEFFECTIVE SAMPLE SIZE:")
    print(f"  GP original (complete cases):     {n_gp_original:,}")
    print(f"  GP matched:                       {len(matched_gp_idx):,} "
          f"({100*len(matched_gp_idx)/n_gp_original:.1f}% retained)")
    print(f"  Comparator original (complete):   {n_comp_original:,}")
    print(f"  Comparator matched:               {len(matched_comp_idx):,} "
          f"({100*len(matched_comp_idx)/n_comp_original:.1f}% used)")

    # Save SMD table for Love plot - uses precomputed values for consistency
    smd_table = pd.DataFrame({
        "variable": SMD_VARS,
        "SMD_before": [smd_before[v] for v in SMD_VARS],
        "SMD_after": [smd_after[v] for v in SMD_VARS],
    })
    smd_table["balanced"] = smd_table["SMD_after"] < 0.1
    smd_table.to_csv("psm_smd_table.csv", index=False)
    print(f"\nWrote psm_smd_table.csv (use for Love plot - SMD before vs after matching)")

print("\nDone. Review SMD values before proceeding to outcome analysis.")
