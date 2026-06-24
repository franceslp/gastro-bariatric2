"""
build_new_funnel.py  

FULL PIPELINE CONTEXT (for reproducibility):
  335,846 gastroparesis patients (K31.84 ever, any date)
    → 2,842 with qualifying bariatric CPT (43775/43644/43645/43846/43847)
    → 1,757 with K31.84 on/after Oct 1 2015, strictly before surgery
    → 1,118 with E10 or E11 ever strictly before surgery  ← START HERE

STARTING POPULATION (1,118):
  Patients with ALL of:
    - Qualifying bariatric CPT code
    - K31.84 on/after Oct 1 2015, ever strictly before surgery (no 1yr limit)
    - E10 or E11 ONLY (not E08/E09/E12/E13) ever strictly before surgery
  The 1yr timing window, GES requirement, and multi-surgery exclusion are
  applied below as exclusion criteria — NOT part of the 1,118 definition.

EXCLUSION CRITERIA (applied sequentially for CONSORT funnel):
  1. Age < 18 at surgery
  2. K31.84 not within 1yr strictly before surgery
     (must be on/after 2015-10-01 AND 1-365 days before bariatric_date)
  3. E10/E11 not within 1yr strictly before surgery (1-365 days)
  4. No GES CPT code on or before surgery date
  5. Multiple bariatric surgeries ever

SENSITIVITY ANALYSIS (not an exclusion):
  Prokinetic within 1yr of K31.84 dx — reference flag only

UPSTREAM ASSUMPTIONS TO VERIFY:
  A. closest_K31_84_strictly_before_surgery was derived from ALL K31.84
     history before surgery with no pre-applied 1yr window. If a 1yr window
     was applied upstream, Funnel 3 will exclude 0 patients (silent error).
     The script prints a warning if this occurs.
  B. closest_E10_E11_date_before_surgery_v2 was generated from E10/E11
     codes ONLY. If E08/E09/E12/E13 were included upstream, the diabetes
     phenotype is broader than intended.
  C. Prokinetic flag = within 365 days AFTER K31.84 diagnosis (not before
     surgery). Confirm upstream before using in sensitivity analysis.
"""

import pandas as pd

print(">>> SCRIPT VERSION: build_new_funnel_v7 <<<")

# ---------------------------------------------------------------------------
# Helper — safe boolean conversion
# ---------------------------------------------------------------------------
def true_flag(series):
    return series.astype(str).str.lower().isin(["true", "1"])

# ---------------------------------------------------------------------------
# Load source files
# ---------------------------------------------------------------------------
f1 = pd.read_csv("funnel_1_all_patients_1118.csv", dtype={"patient_id": str})
ges_src = pd.read_csv("bariatric_patients_K3184_window_check.csv",
                      dtype={"patient_id": str},
                      usecols=["patient_id", "has_GES", "first_GES_date"])

print(f"Base cohort: {len(f1):,} patients")
df = f1.merge(ges_src, on="patient_id", how="left")
assert len(df) == len(f1), "Merge duplicated or lost patients"
assert df["patient_id"].nunique() == len(df), "Duplicate patient_ids introduced by merge"
print(f"Merge check passed ✓")

# ---------------------------------------------------------------------------
# Parse dates and add surgery_year to ALL rows before funnel split
# Fix: surgery_year computed here so every funnel CSV contains it
# ---------------------------------------------------------------------------
df["bariatric_date"] = pd.to_datetime(df["bariatric_date"], errors="coerce")
df["first_GES_date"] = pd.to_datetime(df["first_GES_date"], errors="coerce")
df["surgery_year"]   = df["bariatric_date"].dt.year
OCT_2015 = pd.Timestamp("2015-10-01")

# ---------------------------------------------------------------------------
# Flag 1 — Age < 18
# ---------------------------------------------------------------------------
flag_age = true_flag(df["exclude_age_under_18"])

# ---------------------------------------------------------------------------
# Flag 2 — K31.84 not within 1yr strictly before surgery
# Source note: closest_K31_84_strictly_before_surgery should capture the
# closest K31.84 before surgery from full history with no pre-applied window.
# If upstream already filtered to 1yr, this flag will exclude 0 — see warning.
# ---------------------------------------------------------------------------
df["closest_K3184_dt"] = pd.to_datetime(
    df["closest_K31_84_strictly_before_surgery"], errors="coerce"
)
df["days_K3184_to_surgery"] = (
    df["bariatric_date"] - df["closest_K3184_dt"]
).dt.days

valid_k3184 = (
    (df["closest_K3184_dt"] >= OCT_2015) &
    (df["days_K3184_to_surgery"].between(1, 365))
)
flag_k3184 = ~valid_k3184
print(f"K31.84 flag: {flag_k3184.sum()} excluded")
if flag_k3184.sum() == 0:
    print("  WARNING: K31.84 excludes 0 — upstream may have pre-applied "
          "1yr window. Verify source script.")
print("  K31.84 days-to-surgery distribution:")
print(df["days_K3184_to_surgery"].describe().round(1).to_string())
n_negative_k3184 = (df["days_K3184_to_surgery"] < 0).sum()
n_zero_k3184 = (df["days_K3184_to_surgery"] == 0).sum()
print(f"  K31.84 negative days (should be 0): {n_negative_k3184} "
      f"{'✓' if n_negative_k3184 == 0 else '✗ PROBLEM — upstream extraction wrong'}")
print(f"  K31.84 same-day as surgery (days=0): {n_zero_k3184} "
      f"(excluded by between(1,365))")

# ---------------------------------------------------------------------------
# Flag 3 — E10/E11 not within 1yr strictly before surgery
# Hard assert — no silent fallback
# ---------------------------------------------------------------------------
E10E11_COL = "closest_E10_E11_date_before_surgery_v2"
assert E10E11_COL in df.columns, (
    f"Required column '{E10E11_COL}' not found. "
    f"Available: {[c for c in df.columns if 'E10' in c or 'E11' in c]}"
)

df["closest_E10E11_dt"] = pd.to_datetime(df[E10E11_COL], errors="coerce")
df["days_E10E11_to_surgery"] = (
    df["bariatric_date"] - df["closest_E10E11_dt"]
).dt.days

valid_e10e11 = df["days_E10E11_to_surgery"].between(1, 365)
flag_e10e11  = ~valid_e10e11
print(f"E10/E11 flag: {flag_e10e11.sum()} excluded")
print(f"  NOTE: Verify '{E10E11_COL}' uses E10/E11 only (not E08-E13)")

# ---------------------------------------------------------------------------
# Flag 4 — No GES on or before surgery date
# <= includes same-day GES (appropriate for diagnostic test)
# ---------------------------------------------------------------------------
has_ges          = true_flag(df["has_GES"])
ges_on_or_before = df["first_GES_date"] <= df["bariatric_date"]
flag_no_ges      = ~(has_ges & ges_on_or_before)

ges_missing = (has_ges & df["first_GES_date"].isna()).sum()
print(f"GES True but missing date: {ges_missing} "
      f"{'— INVESTIGATE' if ges_missing > 0 else '✓'}")
print(f"GES flag: {flag_no_ges.sum()} excluded")

# ---------------------------------------------------------------------------
# Flag 5 — Multiple bariatric surgeries
# ---------------------------------------------------------------------------
flag_multi_surgery = true_flag(df["multi_surgery_flag"])
print(f"Multi-surgery flag: {flag_multi_surgery.sum()} excluded")

# ---------------------------------------------------------------------------
# Prokinetic — reference only
# ---------------------------------------------------------------------------
flag_prokinetic = true_flag(df["exclude_no_prokinetic_within_1yr"])

# ---------------------------------------------------------------------------
# Independent exclusion counts
# ---------------------------------------------------------------------------
n = len(df)
print(f"\nExclusion counts (each applied independently to {n:,} patients):")
print(f"  1. Age < 18:                       {flag_age.sum():4d} "
      f"({flag_age.sum()/n*100:.1f}%) → {(~flag_age).sum()} kept")
print(f"  2. K31.84 not within 1yr:          {flag_k3184.sum():4d} "
      f"({flag_k3184.sum()/n*100:.1f}%) → {(~flag_k3184).sum()} kept")
print(f"  3. E10/E11 not within 1yr:         {flag_e10e11.sum():4d} "
      f"({flag_e10e11.sum()/n*100:.1f}%) → {(~flag_e10e11).sum()} kept")
print(f"  4. No GES on/before surgery:       {flag_no_ges.sum():4d} "
      f"({flag_no_ges.sum()/n*100:.1f}%) → {(~flag_no_ges).sum()} kept")
print(f"  5. Multiple bariatric surgeries:   {flag_multi_surgery.sum():4d} "
      f"({flag_multi_surgery.sum()/n*100:.1f}%) → {(~flag_multi_surgery).sum()} kept")
print(f"  [REF] No prokinetic within 1yr:    {flag_prokinetic.sum():4d} "
      f"({flag_prokinetic.sum()/n*100:.1f}%) — sensitivity analysis only")

# ---------------------------------------------------------------------------
# CONSORT funnel (sequential)
# ---------------------------------------------------------------------------
funnel1 = df.copy()
funnel2 = funnel1[~flag_age[funnel1.index]].copy()
funnel3 = funnel2[~flag_k3184[funnel2.index]].copy()
funnel4 = funnel3[~flag_e10e11[funnel3.index]].copy()
funnel5 = funnel4[~flag_no_ges[funnel4.index]].copy()
funnel6 = funnel5[~flag_multi_surgery[funnel5.index]].copy()

print(f"\nCONSORT funnel (sequential):")
print(f"  Funnel 1 — Initial eligible cohort "
      f"(K31.84+DM+bariatric ever): {len(funnel1):4d}")
print(f"  Funnel 2 — After age ≥18:                            "
      f"{len(funnel2):4d} (-{len(funnel1)-len(funnel2)})")
print(f"  Funnel 3 — After K31.84 within 1yr of surgery:      "
      f"{len(funnel3):4d} (-{len(funnel2)-len(funnel3)})")
print(f"  Funnel 4 — After E10/E11 within 1yr of surgery:     "
      f"{len(funnel4):4d} (-{len(funnel3)-len(funnel4)})")
print(f"  Funnel 5 — After GES on/before surgery:             "
      f"{len(funnel5):4d} (-{len(funnel4)-len(funnel5)})")
print(f"  Funnel 6 — After multi-surgery exclusion (FINAL):   "
      f"{len(funnel6):4d} (-{len(funnel5)-len(funnel6)})")

# ---------------------------------------------------------------------------
# Scientific QA — verify final cohort matches phenotype
# ---------------------------------------------------------------------------
passing_mask = (
    ~flag_age &
    ~flag_k3184 &
    ~flag_e10e11 &
    ~flag_no_ges &
    ~flag_multi_surgery
)
qa_df = df.loc[passing_mask, [
    "days_K3184_to_surgery",
    "days_E10E11_to_surgery",
    "first_GES_date",
    "bariatric_date",
    "age_at_surgery_approx",
    "multi_surgery_flag",
]].copy()

print(f"\nScientific QA — timing for patients passing all criteria "
      f"(n={len(qa_df)}):")
print(qa_df[["days_K3184_to_surgery",
             "days_E10E11_to_surgery"]].describe().round(1))

k_ok = qa_df["days_K3184_to_surgery"].between(1, 365).all()
e_ok = qa_df["days_E10E11_to_surgery"].between(1, 365).all()
g_ok = (
    qa_df["first_GES_date"].notna() &
    (qa_df["first_GES_date"] <= qa_df["bariatric_date"])
).all()

# Fix: also report missing GES dates in final cohort
ges_missing_final = qa_df["first_GES_date"].isna().sum()
print(f"\nExpected: K31.84 min≥1 max≤365 | E10/E11 min≥1 max≤365 | GES≤surgery")
print(f"K31.84 timing valid:       {'✓' if k_ok else '✗ PROBLEM'}")
print(f"E10/E11 timing valid:      {'✓' if e_ok else '✗ PROBLEM'}")
print(f"GES on/before surgery:     {'✓' if g_ok else '✗ PROBLEM'}")
print(f"Missing GES dates in final cohort: {ges_missing_final} "
      f"{'— INVESTIGATE' if ges_missing_final > 0 else '✓'}")

# Age check — all should be >= 18
age_col = pd.to_numeric(qa_df["age_at_surgery_approx"], errors="coerce")
age_ok = (age_col >= 18).all()
print(f"All patients age ≥18:      {'✓' if age_ok else '✗ PROBLEM'} "
      f"(min age: {age_col.min():.0f})")

# Multi-surgery check — none should be flagged
multi_ok = (~true_flag(qa_df["multi_surgery_flag"])).all()
print(f"No multi-surgery patients: {'✓' if multi_ok else '✗ PROBLEM'}")

# ---------------------------------------------------------------------------
# Build exclusion criteria sheet
# ---------------------------------------------------------------------------
excl_sheet = df[["patient_id", "bariatric_date", "surgery_year",
                  "age_at_surgery_approx", "sex", "race",
                  "ethnicity"]].copy()
excl_sheet["exclude_age_under_18"]          = flag_age.values
excl_sheet["exclude_K3184_not_within_1yr"]  = flag_k3184.values
excl_sheet["exclude_E10E11_not_within_1yr"] = flag_e10e11.values
excl_sheet["exclude_no_GES_before_surgery"] = flag_no_ges.values
excl_sheet["exclude_multi_surgery"]         = flag_multi_surgery.values
excl_sheet["prokinetic_sensitivity_ref"]    = flag_prokinetic.values
excl_sheet["passes_all_criteria"]           = (
    ~flag_age & ~flag_k3184 & ~flag_e10e11 &
    ~flag_no_ges & ~flag_multi_surgery
).values
excl_sheet["num_exclusion_reasons"] = (
    flag_age.astype(int) + flag_k3184.astype(int) +
    flag_e10e11.astype(int) + flag_no_ges.astype(int) +
    flag_multi_surgery.astype(int)
).values
excl_sheet["days_K3184_to_surgery"]  = df["days_K3184_to_surgery"].values
excl_sheet["days_E10E11_to_surgery"] = df["days_E10E11_to_surgery"].values
excl_sheet["first_GES_date"]         = df["first_GES_date"].values
excl_sheet["has_GES"]                = df["has_GES"].values

assert excl_sheet["passes_all_criteria"].sum() == len(funnel6), (
    f"Mismatch: sheet={excl_sheet['passes_all_criteria'].sum()} "
    f"funnel6={len(funnel6)}"
)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
funnel1.to_csv("funnel_1_new.csv", index=False)
funnel2.to_csv("funnel_2_new.csv", index=False)
funnel3.to_csv("funnel_3_new.csv", index=False)
funnel4.to_csv("funnel_4_new.csv", index=False)
funnel5.to_csv("funnel_5_new.csv", index=False)
funnel6.to_csv("funnel_6_new.csv", index=False)
excl_sheet.to_csv("exclusion_criteria_sheet_new.csv", index=False)

print(f"\nFiles written: funnel_1_new.csv → funnel_6_new.csv")
print(f"               exclusion_criteria_sheet_new.csv ({len(excl_sheet):,} rows)")

# ---------------------------------------------------------------------------
# Final cohort summary
# ---------------------------------------------------------------------------
denom = max(len(funnel6), 1)
print(f"\nFINAL COHORT SUMMARY (n={len(funnel6)}):")

if "sex" in funnel6.columns:
    print(f"  Sex: {funnel6['sex'].value_counts().to_dict()}")
if "race" in funnel6.columns:
    print(f"  Race (top 3): "
          f"{funnel6['race'].value_counts().head(3).to_dict()}")

cpt = (funnel6["bariatric_cpt_codes_seen"]
       .astype(str).str.replace(r"\.0", "", regex=True))
sleeve = cpt.str.contains("43775", na=False)
bypass = cpt.str.contains("43644|43645|43846|43847", na=False, regex=True)
both   = sleeve & bypass
print(f"\n  Surgery type:")
print(f"    Sleeve only:  {(sleeve & ~bypass).sum():3d} "
      f"({(sleeve & ~bypass).sum()/denom*100:.1f}%)")
print(f"    Bypass only:  {(bypass & ~sleeve).sum():3d} "
      f"({(bypass & ~sleeve).sum()/denom*100:.1f}%)")
print(f"    Both codes:   {both.sum():3d} "
      f"({both.sum()/denom*100:.1f}%)")

prok_final = flag_prokinetic.loc[funnel6.index]
prok_yes = (~prok_final).sum()
prok_no  = prok_final.sum()
print(f"\n  Prokinetic use (sensitivity analysis reference):")
print(f"    With prokinetic:    {prok_yes:3d} ({prok_yes/denom*100:.1f}%)")
print(f"    Without prokinetic: {prok_no:3d} ({prok_no/denom*100:.1f}%)")

print(f"\n  Surgery year distribution:")
print(funnel6["surgery_year"].value_counts().sort_index().to_string())

print("\nDone.")
