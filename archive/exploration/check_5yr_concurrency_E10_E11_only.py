"""
check_5yr_concurrency_E10_E11_only.py

Replicates Sadda et al.'s 5-year concurrency rule, but restricts the
diabetes signal to E10 (Type 1) and E11 (Type 2) only - NOT the full
E08-E13 block Sadda actually used. This is a deliberate methodological
deviation, not a literal replication; see chat for why Sadda's original
rule used the broader diabetes block, not just T1/T2.

WHAT SADDA'S RULE ACTUALLY DOES (from their published Methods text):
  "the first concurrent documentation of diabetes and gastroparesis was
   required to occur within 5 years before the start of follow-up"
  i.e. max(first_diabetes_date, first_K31_84_date) must fall within 5
  years (here: 1826 days, round(5*365.25)) before the surgery date.
  NEITHER date individually has a window - only the LATER of the two
  needs to be recent relative to surgery. The earlier one can be
  arbitrarily old. This script preserves that exact mechanic, just with
  a narrower diabetes definition.

NO NEW SCAN NEEDED: first_E10_date and first_E11_date are already
produced by check_E10_E11_prior_to_K3184_bariatric_subset.py for this
exact patient population (the bariatric-surgery subset). This script is
pure arithmetic on that output - run check_E10_E11_prior_to_K3184_bariatric_subset.py
FIRST if you haven't already.

first_E10_or_E11_date = the earlier of first_E10_date and first_E11_date
(whichever exists / is earlier) - this mirrors how Sadda's diabetes date
itself has no window, just "earliest ever documented."

ALSO ADDS diagnosis order columns: which condition came first, gastroparesis
or E10/E11 diabetes. Purely descriptive - the concurrency rule itself
doesn't care about order (only whichever date is LATER matters, regardless
of which condition it belongs to), but the order is a separate, interesting
question worth tracking on its own.

Run on the bariatric_subset_with_E10_E11_timing.csv output. Writes a NEW
file.
"""

import pandas as pd

INPUT_CSV = "bariatric_subset_with_E10_E11_timing.csv"
OUTPUT_CSV = "bariatric_subset_with_5yr_concurrency_E10_E11.csv"

WINDOW_DAYS = round(5 * 365.25)  # 1826, matching the original Sadda 5yr script

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = ["first_E10_date", "first_E11_date", "first_K31_84_date", "bariatric_date", "in_study_period"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns: {missing} - this script needs the output of "
        f"check_E10_E11_prior_to_K3184_bariatric_subset.py as input, not the original "
        f"bariatric-and-concurrency file."
    )

first_E10_dt = pd.to_datetime(df["first_E10_date"], errors="coerce")
first_E11_dt = pd.to_datetime(df["first_E11_date"], errors="coerce")
first_k3184_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
bariatric_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")

# Earliest of E10 or E11 (whichever exists / is earlier) - no window on
# this date itself, matching how Sadda's own diabetes date has no window.
first_E10_or_E11_dt = pd.concat([first_E10_dt, first_E11_dt], axis=1).min(axis=1, skipna=True)
df["first_E10_or_E11_date"] = first_E10_or_E11_dt

both_dx_present = first_E10_or_E11_dt.notna() & first_k3184_dt.notna()
# Same .where() pattern as the original Sadda script - .max(axis=1) alone
# would return whichever date exists even if the OTHER is NaT, which isn't
# "concurrent" since that requires BOTH to be present.
raw_concurrent_dt = pd.concat([first_k3184_dt, first_E10_or_E11_dt], axis=1).max(axis=1)
concurrent_dt = raw_concurrent_dt.where(both_dx_present, pd.NaT)
df["E10_E11_gastroparesis_concurrent_date"] = concurrent_dt

all_required_present = both_dx_present & bariatric_dt.notna()
df["days_E10_E11_concurrent_to_bariatric_surgery"] = (bariatric_dt - concurrent_dt).dt.days

df["meets_5yr_concurrency_rule_E10_E11_only"] = (
    all_required_present
    & (df["days_E10_E11_concurrent_to_bariatric_surgery"] >= 0)
    & (df["days_E10_E11_concurrent_to_bariatric_surgery"] <= WINDOW_DAYS)
)

# --- Diagnosis order: which came first, gastroparesis or diabetes (E10/E11)? ---
# Purely descriptive - the concurrency rule above doesn't care about order
# (only the LATER date matters, regardless of which condition it belongs
# to), but the order itself is a separate, interesting question on its own.
# Only meaningful when both dates are present - patients missing either
# date get "missing_one_or_both" rather than a forced guess.
days_gp_minus_diabetes = (first_k3184_dt - first_E10_or_E11_dt).dt.days

diagnosis_order = pd.Series("missing_one_or_both", index=df.index, dtype=object)
diagnosis_order[both_dx_present & (days_gp_minus_diabetes < 0)] = "gastroparesis_first"
diagnosis_order[both_dx_present & (days_gp_minus_diabetes > 0)] = "diabetes_first"
diagnosis_order[both_dx_present & (days_gp_minus_diabetes == 0)] = "same_day"
df["E10_E11_diagnosis_order"] = diagnosis_order

# Convenience booleans, same information as above, easier to filter/sum on.
df["gastroparesis_diagnosed_before_E10_E11"] = diagnosis_order == "gastroparesis_first"
df["E10_E11_diagnosed_before_gastroparesis"] = diagnosis_order == "diabetes_first"
df["gastroparesis_and_E10_E11_same_day"] = diagnosis_order == "same_day"

print("\nSANITY CHECK:")
in_period = df["in_study_period"]
n_total = in_period.sum()
print(f"\nOf the {n_total:,} in-study-period K31.84 bariatric-surgery patients:")
print(f"  have E10 or E11 documented (any time):                  {(in_period & first_E10_or_E11_dt.notna()).sum():,}")
print(f"  meet 5yr concurrency rule (E10/E11 only):                {(in_period & df['meets_5yr_concurrency_rule_E10_E11_only']).sum():,}")

if "meets_5yr_concurrency_rule" in df.columns:
    print(f"\n  (for comparison) meets 5yr concurrency rule (E08-E13, Sadda's actual rule): "
          f"{(in_period & df['meets_5yr_concurrency_rule']).sum():,}")
else:
    print("\n  (NOTE: meets_5yr_concurrency_rule [E08-E13 version] column not found in this file - "
          "can't print a side-by-side comparison. That column lives in the original "
          "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv file.)")

print(f"\nDiagnosis order (of patients with BOTH dates present, i.e. {(in_period & both_dx_present).sum():,} patients):")
print(f"  gastroparesis diagnosed first:    {(in_period & df['gastroparesis_diagnosed_before_E10_E11']).sum():,}")
print(f"  E10/E11 diabetes diagnosed first: {(in_period & df['E10_E11_diagnosed_before_gastroparesis']).sum():,}")
print(f"  same day:                          {(in_period & df['gastroparesis_and_E10_E11_same_day']).sum():,}")
print(f"  (missing one or both dates, excluded from above): {(in_period & ~both_dx_present).sum():,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with first_E10_or_E11_date, E10_E11_gastroparesis_concurrent_date,")
print("days_E10_E11_concurrent_to_bariatric_surgery, meets_5yr_concurrency_rule_E10_E11_only,")
print("E10_E11_diagnosis_order, gastroparesis_diagnosed_before_E10_E11,")
print("E10_E11_diagnosed_before_gastroparesis, and gastroparesis_and_E10_E11_same_day columns.")
print(f"(Original {INPUT_CSV} left untouched.)")
