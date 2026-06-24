"""
build_master_cohort_new.py  

Step 2 of the revised pipeline.
Reads funnel_6_new.csv (produced by build_new_funnel.py) and builds:

  1. master_cohort_FINAL_new.csv — clean 28-column master reference file
     (one row per patient in final cohort, no exclusion flags)

  2. Validates exclusion_criteria_sheet_new.csv written by build_new_funnel.py

STUDY PHENOTYPE (important for manuscript methods):
  "Patients with documented active gastroparesis (K31.84 within 1 year) AND
  active diabetes (E10/E11 within 1 year) preceding bariatric surgery, with
  objective gastric emptying scintigraphy confirmed before surgery, and no
  prior bariatric surgical history."

Column changes vs original master_cohort_FINAL_1118.csv:
  REMOVED:
    closest_GES_before_K3184_dx_code (no longer windowed to K31.84)
    closest_GES_before_K3184_dx_date (replaced by first_GES_date)
    days_GES_before_K3184            (replaced by days_GES_to_surgery)

  ADDED:
    surgery_year         — calendar year of bariatric surgery
    surgery_type         — sleeve / bypass / both (derived from CPT codes)
    has_GES              — boolean, True = GES confirmed before surgery
    first_GES_date       — date of first GES ever before surgery
    days_GES_to_surgery  — days between first GES and bariatric surgery
    multi_surgery_flag   — reference only (all False in final cohort)

UPSTREAM ASSUMPTIONS:
  - diabetes_type_label was generated from E10/E11 codes only. If it
    includes E08-E13, the label does not match the funnel definition.
    Script prints value_counts() so you can verify.
  - days_to_prokinetic_after_K3184 = days after K31.84 dx (not before
    surgery). Script prints describe() so you can verify range.
"""

import pandas as pd

print(">>> SCRIPT VERSION: build_master_cohort_new_v3 <<<")

# ---------------------------------------------------------------------------
# Load final cohort from Step 1 output
# ---------------------------------------------------------------------------
funnel6 = pd.read_csv("funnel_6_new.csv", dtype={"patient_id": str})
print(f"Final cohort loaded: {len(funnel6):,} patients")

# Load GES source
ges_src = pd.read_csv("bariatric_patients_K3184_window_check.csv",
                      dtype={"patient_id": str},
                      usecols=["patient_id", "has_GES", "first_GES_date",
                               "num_K31_84_encounters", "K3184_span_days",
                               "all_K31_84_dates"])

print("Merging GES and K31.84 history...")
df = funnel6.merge(ges_src, on="patient_id", how="left", suffixes=("", "_ges"))

assert len(df) == len(funnel6), \
    f"Merge changed row count: {len(funnel6)} → {len(df)}"
assert df["patient_id"].nunique() == len(df), \
    "Duplicate patient_ids after merge"
print(f"Merge check passed ✓ ({len(df):,} rows)")

# ---------------------------------------------------------------------------
# Parse dates
# ---------------------------------------------------------------------------
df["bariatric_date"] = pd.to_datetime(df["bariatric_date"], errors="coerce")
df["first_GES_date"] = pd.to_datetime(df["first_GES_date"], errors="coerce")

# ---------------------------------------------------------------------------
# Derive columns
# ---------------------------------------------------------------------------
df["surgery_year"] = df["bariatric_date"].dt.year

# Surgery type
cpt = df["bariatric_cpt_codes_seen"].astype(str).str.replace(r"\.0", "", regex=True)
sleeve = cpt.str.contains("43775", na=False)
bypass = cpt.str.contains("43644|43645|43846|43847", na=False, regex=True)
df["surgery_type"] = "unknown"
df.loc[sleeve & ~bypass, "surgery_type"] = "sleeve"
df.loc[bypass & ~sleeve, "surgery_type"] = "bypass"
df.loc[sleeve & bypass,  "surgery_type"] = "both"

# FIX 2 — has_GES as boolean (NaN → False, then assert all True)
df["has_GES"] = df["has_GES"].astype(str).str.lower().isin(["true", "1"])

# FIX 4 — days from first GES to surgery (reviewers will ask this)
df["days_GES_to_surgery"] = (
    df["bariatric_date"] - df["first_GES_date"]
).dt.days

# Resolve column duplication from merge
for col in ["num_K31_84_encounters", "K3184_span_days", "all_K31_84_dates"]:
    ges_col = f"{col}_ges"
    if ges_col in df.columns:
        if col not in df.columns or df[col].isna().all():
            df[col] = df[ges_col]
        df = df.drop(columns=[ges_col])

# ---------------------------------------------------------------------------
# Build master — 27 clean columns, no exclusion flags
# FIX 1 — header says 27 cols (26 original redesigned + days_GES_to_surgery)
# FIX 6 — fixed output name master_cohort_FINAL_new.csv
# ---------------------------------------------------------------------------
MASTER_COLS = [
    "patient_id",
    "year_of_birth",
    "age_at_surgery_approx",
    "bariatric_date",
    "surgery_year",
    "bariatric_cpt_codes_seen",
    "surgery_type",
    "multi_surgery_flag",
    "closest_K31_84_strictly_before_surgery",
    "num_K31_84_encounters",
    "K3184_span_days",
    "all_K31_84_dates",
    "diabetes_type_label",
    "closest_E10_E11_code_before_surgery",
    "closest_E10_E11_date_before_surgery",
    "has_GES",
    "first_GES_date",
    "days_GES_to_surgery",
    "first_prokinetic_after_K3184_dx_drug",
    "first_prokinetic_after_K3184_dx_date",
    "days_to_prokinetic_after_K3184",
    "sex",
    "race",
    "ethnicity",
    "marital_status",
    "month_year_death",
    "deceased",
    "meets_age_requirement",
]

missing_cols = [c for c in MASTER_COLS if c not in df.columns]
if missing_cols:
    print(f"\nWARNING: Columns not found in source — will be NaN:")
    for c in missing_cols:
        print(f"  {c}")

available_cols = [c for c in MASTER_COLS if c in df.columns]
master = df[available_cols].copy()
for c in missing_cols:
    master[c] = pd.NA
master = master[MASTER_COLS]

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
n = len(master)
print(f"\nMaster cohort QA:")
print(f"  Rows:    {n:,}")
print(f"  Columns: {len(master.columns)} (expected: {len(MASTER_COLS)})")
print(f"  All columns present: {len(missing_cols) == 0}")
print(f"  Duplicate patient_ids: {master['patient_id'].duplicated().sum()}")

# FIX 2 — assert GES confirmed for all patients (hard stop not just print)
assert master["has_GES"].all(), \
    "CRITICAL: Final cohort contains patients without confirmed GES — " \
    "inclusion criteria violated"
print(f"  has_GES = True for all {n} patients ✓")

# FIX 1 — assert GES date present for all (has_GES=True but NaT would slip through)
assert master["first_GES_date"].notna().all(), \
    "CRITICAL: has_GES=True but first_GES_date is missing for some patients"
print(f"  first_GES_date present for all {n} patients ✓")

# GES timing
print(f"\n  GES timing (days_GES_to_surgery):")
print(master["days_GES_to_surgery"].describe().round(1).to_string())
# FIX 2 — assert GES before surgery (hard stop not just print)
assert (master["days_GES_to_surgery"] >= 0).all(), \
    "CRITICAL: GES occurred after bariatric surgery — phenotype violated"
print(f"  All GES on or before surgery ✓")

# Surgery type
print(f"\n  Surgery type:")
print(master["surgery_type"].value_counts().to_string())

# Age
age = pd.to_numeric(master["age_at_surgery_approx"], errors="coerce")
print(f"\n  Age: min={age.min():.0f}, max={age.max():.0f}, "
      f"mean={age.mean():.1f}")
assert (age >= 18).all(), "Patient under 18 found in final cohort"
print(f"  All patients age ≥18 ✓")

# Surgery year
print(f"\n  Surgery year distribution:")
print(master["surgery_year"].value_counts().sort_index().to_string())

# Sex
print(f"\n  Sex: {master['sex'].value_counts().to_dict()}")

# FIX 3 — diabetes type label check (flag if non-E10/E11 types present)
print(f"\n  Diabetes type label (verify E10/E11 only):")
print(master["diabetes_type_label"].value_counts(dropna=False).to_string())
suspicious = master["diabetes_type_label"].astype(str).str.lower().str.contains(
    "other|secondary|e08|e09|e12|e13", na=False
).sum()
if suspicious > 0:
    print(f"  WARNING: {suspicious} patients have non-E10/E11 diabetes labels "
          f"— verify upstream code definition")
else:
    print(f"  No non-E10/E11 labels detected ✓")

# FIX 5 — prokinetic timing validation
print(f"\n  Prokinetic days-after-K31.84 (sensitivity analysis reference):")
prok_days = pd.to_numeric(
    master["days_to_prokinetic_after_K3184"], errors="coerce"
)
print(prok_days.describe().round(1).to_string())
prok_neg = (prok_days < 0).sum()
prok_over365 = (prok_days > 365).sum()
prok_missing = prok_days.isna().sum()
print(f"  Negative (before K31.84): {prok_neg} "
      f"{'✓' if prok_neg == 0 else '✗ CHECK'}")
print(f"  Over 365 days:            {prok_over365} "
      f"(these are outside 1yr window — excluded from sensitivity analysis)")
print(f"  Missing (no prokinetic):  {prok_missing} "
      f"({prok_missing/n*100:.1f}%)")

# Validate exclusion sheet
try:
    excl = pd.read_csv("exclusion_criteria_sheet_new.csv",
                       dtype={"patient_id": str})
    passing = excl["passes_all_criteria"].sum()
    match = passing == n
    print(f"\nExclusion sheet validation:")
    print(f"  passes_all_criteria = {passing} | master rows = {n} "
          f"{'✓' if match else '✗ MISMATCH — rerun build_new_funnel.py'}")
except FileNotFoundError:
    print("\nWARNING: exclusion_criteria_sheet_new.csv not found "
          "— run build_new_funnel.py first")

# Column order
print(f"\nColumn order ({len(master.columns)} cols):")
for i, col in enumerate(master.columns, 1):
    flag = " ← MISSING UPSTREAM" if col in missing_cols else ""
    print(f"  {i:2d}. {col}{flag}")

# ---------------------------------------------------------------------------
# Save — FIX 6: fixed filename, print n separately
# ---------------------------------------------------------------------------
OUTPUT = "master_cohort_FINAL_new.csv"
master.to_csv(OUTPUT, index=False)
print(f"\nWrote {OUTPUT} (n={n:,}, {len(master.columns)} cols)")
print("Done.")
