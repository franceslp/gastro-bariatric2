"""
check_prokinetic_and_GES_intersection.py

Quick filter on the existing master cohort file - no GCS scan, just
boolean logic on columns that already exist side by side. Computes the
overlap between prokinetic drug evidence and GES evidence under a few
reasonable variants of each, since "prokinetic" and "GES" each have more
than one definition already in use in this pipeline.

Prokinetic variants already in the file:
  - any_prokinetic_ever_after_dx  (at least one of the 4 drugs, dated on/
    after first K31.84 dx)
  - any_prokinetic_any_time       (at least one of the 4 drugs, anywhere
    in the patient's record, no timing requirement)

GES variants computed here from first_GES_date vs first_K31_84_date
(arithmetic only, not a rescan):
  - GES any time at all            (has_GES)
  - GES before dx (strictly)
  - GES before-or-same-day as dx

This intentionally does NOT use a +/-90-day "near dx" window for GES -
that's a separate, narrower question. This script is just: of the people
who have BOTH kinds of corroborating evidence, regardless of exactly when
the GES happened, how many are there.

Run on the VM (or anywhere with the CSV) - this is instant, no nohup needed.
"""

import pandas as pd

INPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

required_cols = ["first_K31_84_date", "first_GES_date", "has_GES", "any_prokinetic_ever_after_dx", "any_prokinetic_any_time", "in_study_period"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing} - check you're pointing at the right input file.")

dx_date = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_date = pd.to_datetime(df["first_GES_date"], errors="coerce")

df["ges_before_dx"] = ges_date.notna() & dx_date.notna() & (ges_date < dx_date)
df["ges_before_or_same_day_dx"] = ges_date.notna() & dx_date.notna() & (ges_date <= dx_date)

in_period = df["in_study_period"]
n_total = in_period.sum()
print(f"\nOf the {n_total:,} in-study-period K31.84 patients:\n")

combos = [
    ("prokinetic ever-after-dx  AND  GES any time",            df["any_prokinetic_ever_after_dx"] & df["has_GES"]),
    ("prokinetic ever-after-dx  AND  GES before dx",            df["any_prokinetic_ever_after_dx"] & df["ges_before_dx"]),
    ("prokinetic ever-after-dx  AND  GES before-or-same-day",   df["any_prokinetic_ever_after_dx"] & df["ges_before_or_same_day_dx"]),
    ("prokinetic any-time       AND  GES any time",             df["any_prokinetic_any_time"] & df["has_GES"]),
    ("prokinetic any-time       AND  GES before dx",            df["any_prokinetic_any_time"] & df["ges_before_dx"]),
    ("prokinetic any-time       AND  GES before-or-same-day",   df["any_prokinetic_any_time"] & df["ges_before_or_same_day_dx"]),
]

for label, mask in combos:
    n = (in_period & mask).sum()
    pct = 100 * n / n_total
    print(f"  {label}: {n:,} ({pct:.1f}%)")

print("\n(For reference, the individual components alone:)")
print(f"  prokinetic ever-after-dx alone: {(in_period & df['any_prokinetic_ever_after_dx']).sum():,}")
print(f"  prokinetic any-time alone:      {(in_period & df['any_prokinetic_any_time']).sum():,}")
print(f"  GES any time alone:             {(in_period & df['has_GES']).sum():,}")
print(f"  GES before dx alone:            {(in_period & df['ges_before_dx']).sum():,}")
print(f"  GES before-or-same-day alone:   {(in_period & df['ges_before_or_same_day_dx']).sum():,}")
