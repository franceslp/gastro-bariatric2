"""
run_PSM_no_BMI.py

PSM WITHOUT BMI covariate - uses all 110 gastroparesis patients.

Identical to run_PSM.py except BMI is excluded from covariates,
allowing all 110 GP patients (not just the 76 with pre-op BMI) to
be matched. Present both versions to Dr. Sujka to decide which to use.

PSM covariates (5, no BMI):
  - age_at_surgery_approx
  - sex
  - surgery_type (sleeve vs bypass)
  - has_diabetes
  - surgery_year (one-hot)

Outputs (separate filenames to avoid overwriting BMI version):
  psm_matched_pairs_no_BMI.csv
  psm_all_candidates_with_scores_no_BMI.csv
  psm_propensity_scores_by_group_no_BMI.csv
  psm_smd_table_no_BMI.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GP_CSV = "funnel_6_final_cohort.csv"
COMPARATOR_CSV = "comparator_pool_ready_for_PSM.csv"  # without BMI - uses full 6,806 pool
OUTPUT_MATCHED = "psm_matched_pairs_no_BMI.csv"
OUTPUT_ALL_SCORES = "psm_all_candidates_with_scores_no_BMI.csv"

CALIPER_MULTIPLIER = 0.2
RANDOM_STATE = 42

SLEEVE_CODES = {"43775"}
BYPASS_CODES = {"43644", "43645", "43846", "43847"}

print(">>> SCRIPT VERSION: run_PSM_no_BMI_v1 <<<")
print("NOTE: BMI excluded - matching all 110 GP patients")

gp = pd.read_csv(GP_CSV, dtype={"patient_id": str}, low_memory=False,
                  usecols=["patient_id", "bariatric_date", "sex",
                            "bariatric_cpt_codes_seen", "diabetes_type_label",
                            "age_at_surgery_approx", "year_of_birth"])

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
print(f"Study group: {len(gp):,} patients")

comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str}, low_memory=False)
comp["group"] = 0
print(f"Comparator pool: {len(comp):,} patients")

BASE_VARS = ["patient_id", "group", "age_at_surgery_approx", "sex",
             "surgery_type", "has_diabetes", "surgery_year"]

combined = pd.concat([gp[BASE_VARS], comp[BASE_VARS]], ignore_index=True)

print(f"\nMissingness per PSM covariate:")
print((combined[["age_at_surgery_approx", "sex", "surgery_type",
                  "has_diabetes", "surgery_year"]].isna().mean() * 100
       ).round(1).sort_values(ascending=False))

n_before = len(combined)
combined = combined.dropna(subset=["age_at_surgery_approx", "sex", "surgery_type",
                                    "has_diabetes", "surgery_year"])
print(f"Dropped {n_before-len(combined):,} with missing covariates")
print(f"  Study group: {(combined['group']==1).sum():,}")
print(f"  Comparator:  {(combined['group']==0).sum():,}")

assert combined["surgery_type"].isin(["sleeve", "bypass"]).all()

combined["sex_encoded"] = (combined["sex"].str.upper() == "F").astype(int)
combined["surgery_type_encoded"] = (combined["surgery_type"] == "sleeve").astype(int)
combined["has_diabetes_encoded"] = combined["has_diabetes"].fillna(False).astype(int)

year_dummies = pd.get_dummies(combined["surgery_year"].astype(int),
                               prefix="yr", drop_first=True)
combined = pd.concat([combined.reset_index(drop=True),
                       year_dummies.reset_index(drop=True)], axis=1)
year_cols = [c for c in combined.columns if c.startswith("yr_")]

NUMERIC_VARS = (["age_at_surgery_approx", "sex_encoded", "surgery_type_encoded",
                  "has_diabetes_encoded"] + year_cols)

X = combined[NUMERIC_VARS].values.astype(float)
y = combined["group"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFitting logistic regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_scaled, y)
combined["propensity_score"] = lr.predict_proba(X_scaled)[:, 1]
ps_clipped = combined["propensity_score"].clip(1e-6, 1 - 1e-6)
combined["logit_ps"] = np.log(ps_clipped / (1 - ps_clipped))

print(f"PS mean - GP: {combined.loc[combined['group']==1,'propensity_score'].mean():.3f}, "
      f"Comp: {combined.loc[combined['group']==0,'propensity_score'].mean():.3f}")

combined.to_csv(OUTPUT_ALL_SCORES, index=False)

def smd(var, df, group_col="group"):
    g1 = df.loc[df[group_col]==1, var].dropna()
    g0 = df.loc[df[group_col]==0, var].dropna()
    if var.endswith("_encoded"):
        return abs(g1.mean() - g0.mean())
    pooled_sd = np.sqrt((g1.std()**2 + g0.std()**2) / 2)
    return 0.0 if pooled_sd == 0 else abs(g1.mean() - g0.mean()) / pooled_sd

SMD_VARS = ["age_at_surgery_approx", "sex_encoded",
            "surgery_type_encoded", "has_diabetes_encoded"]
smd_before_matching = {v: smd(v, combined) for v in SMD_VARS}

print("\nSMD BEFORE matching:")
for v in SMD_VARS:
    print(f"  {v}: {smd_before_matching[v]:.3f}")

# Overlap
gp_ps = combined.loc[combined["group"]==1, "propensity_score"]
comp_ps = combined.loc[combined["group"]==0, "propensity_score"]
overlap_min = max(gp_ps.min(), comp_ps.min())
overlap_max = min(gp_ps.max(), comp_ps.max())
gp_outside = ((gp_ps < comp_ps.min()) | (gp_ps > comp_ps.max())).sum()
print(f"\nOverlap: [{overlap_min:.3f}, {overlap_max:.3f}]")
print(f"GP outside comparator support: {gp_outside:,}/{len(gp_ps):,}")

combined.to_csv("psm_propensity_scores_by_group_no_BMI.csv", index=False)

caliper = CALIPER_MULTIPLIER * combined["logit_ps"].std()
print(f"\nLogit PS SD: {combined['logit_ps'].std():.4f}")
print(f"Caliper: {caliper:.4f}")

gp_pool = combined[combined["group"]==1].copy().reset_index(drop=True)
comp_pool = combined[combined["group"]==0].copy().reset_index(drop=True)
gp_pool = gp_pool.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

matched_gp_idx, matched_comp_idx, unmatched_gp = [], [], []
used_comp_indices = set()

for i, gp_row in gp_pool.iterrows():
    available = comp_pool[~comp_pool.index.isin(used_comp_indices)]
    if available.empty:
        unmatched_gp.append(gp_row["patient_id"])
        continue
    eligible = available.loc[(available["logit_ps"] - gp_row["logit_ps"]).abs() <= caliper]
    if eligible.empty:
        unmatched_gp.append(gp_row["patient_id"])
        continue
    best_idx = (eligible["logit_ps"] - gp_row["logit_ps"]).abs().idxmin()
    matched_gp_idx.append(i)
    matched_comp_idx.append(best_idx)
    used_comp_indices.add(best_idx)

print(f"\nMatching results:")
print(f"  Matched pairs:         {len(matched_gp_idx):,}")
print(f"  Unmatched GP patients: {len(unmatched_gp):,}")
if unmatched_gp:
    print(f"  Unmatched IDs: {unmatched_gp}")

if matched_gp_idx:
    matched_combined = pd.concat([
        gp_pool.loc[matched_gp_idx].assign(group=1),
        comp_pool.loc[matched_comp_idx].assign(group=0)
    ], ignore_index=True)

    smd_after = {v: smd(v, matched_combined) for v in SMD_VARS}
    print("\nSMD AFTER matching:")
    for v in SMD_VARS:
        flag = " ✓" if smd_after[v] < 0.1 else " ← REVIEW"
        print(f"  {v}: {smd_after[v]:.3f}{flag}")

    n_gp_orig = (combined["group"]==1).sum()
    n_comp_orig = (combined["group"]==0).sum()
    print(f"\nEffective sample size:")
    print(f"  GP matched:         {len(matched_gp_idx):,}/{n_gp_orig:,} "
          f"({100*len(matched_gp_idx)/n_gp_orig:.1f}%)")
    print(f"  Comparator matched: {len(matched_comp_idx):,}/{n_comp_orig:,} "
          f"({100*len(matched_comp_idx)/n_comp_orig:.1f}%)")

    gp_matched = gp_pool.loc[matched_gp_idx].add_suffix("_gp").reset_index(drop=True)
    comp_matched = comp_pool.loc[matched_comp_idx].add_suffix("_comp").reset_index(drop=True)
    pairs = pd.concat([gp_matched, comp_matched], axis=1)
    pairs["pair_id"] = range(1, len(pairs)+1)
    pairs.to_csv(OUTPUT_MATCHED, index=False)
    print(f"\nWrote {OUTPUT_MATCHED}")

    smd_table = pd.DataFrame({
        "variable": SMD_VARS,
        "SMD_before": [smd_before_matching[v] for v in SMD_VARS],
        "SMD_after": [smd_after[v] for v in SMD_VARS],
    })
    smd_table["balanced"] = smd_table["SMD_after"] < 0.1
    smd_table.to_csv("psm_smd_table_no_BMI.csv", index=False)
    print(f"Wrote psm_smd_table_no_BMI.csv")

print("\nDone.")
