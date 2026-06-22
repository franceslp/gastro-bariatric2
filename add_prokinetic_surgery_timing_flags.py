"""
add_prokinetic_surgery_timing_flags.py

Adds two flag columns to the prokinetic output file:
  prokinetic_same_day_as_surgery: True if prokinetic start date == surgery date
  prokinetic_after_surgery: True if prokinetic start date > surgery date

These let Dr. Sujka decide how to handle each group - same-day prokinetics
in a diabetic gastroparesis cohort may reflect active peri-operative
gastroparesis management rather than routine post-bariatric protocol.

No new scan - pure arithmetic on existing columns.
"""

import pandas as pd

INPUT_CSV = "prokinetic_timing_vs_surgery_check.csv"
OUTPUT_CSV = "cohort_prokinetic_with_surgery_timing_flags.csv"

print(">>> SCRIPT VERSION: add_prokinetic_surgery_timing_flags_v1 <<<")

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Cohort size: {len(df):,} patients")

prok_dt = pd.to_datetime(df["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
surgery_dt = pd.to_datetime(df["bariatric_date"], errors="coerce")

has_prokinetic = prok_dt.notna()
days_prok_to_surgery = (surgery_dt - prok_dt).dt.days

df["prokinetic_same_day_as_surgery"] = has_prokinetic & (days_prok_to_surgery == 0)
df["prokinetic_after_surgery"] = has_prokinetic & (days_prok_to_surgery < 0)
df["prokinetic_before_surgery"] = has_prokinetic & (days_prok_to_surgery > 0)

n_with = has_prokinetic.sum()
n_same = df["prokinetic_same_day_as_surgery"].sum()
n_after = df["prokinetic_after_surgery"].sum()
n_before = df["prokinetic_before_surgery"].sum()

print(f"\nOf {n_with:,} patients with a qualifying prokinetic:")
print(f"  Before surgery:      {n_before:,} ({100*n_before/n_with:.1f}%)")
print(f"  Same day as surgery: {n_same:,} ({100*n_same/n_with:.1f}%)")
print(f"  After surgery:       {n_after:,} ({100*n_after/n_with:.1f}%)")
print(f"\nAs % of full cohort ({len(df):,} patients):")
print(f"  Before surgery:      {n_before:,} ({100*n_before/len(df):.1f}%)")
print(f"  Same day as surgery: {n_same:,} ({100*n_same/len(df):.1f}%)")
print(f"  After surgery:       {n_after:,} ({100*n_after/len(df):.1f}%)")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
