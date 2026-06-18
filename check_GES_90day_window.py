"""
check_GES_90day_window.py

Quick filter on the existing master cohort file - no GCS scan. Computes
how many patients had their (first recorded) GES within 90 days before
their K31.84 diagnosis, using columns that already exist in the file.

CAVEAT: this only checks each patient's FIRST recorded GES date against
the window. A patient whose first-ever GES falls outside 90 days, but who
had a LATER GES that actually falls within the window, won't be caught
here - only first/last GES dates were ever saved, not every individual
occurrence. Probably a minor effect since GES isn't typically repeated
the way a chronic medication is, but worth knowing.

Run on the VM (or anywhere with the CSV) - instant, no nohup needed.
"""

import pandas as pd

INPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"
WINDOW_DAYS = 90

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)

dx_date = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_date = pd.to_datetime(df["first_GES_date"], errors="coerce")

days_before_dx = (dx_date - ges_date).dt.days
df["ges_within_90d_before_dx"] = (
    ges_date.notna() & dx_date.notna() & (days_before_dx >= 0) & (days_before_dx <= WINDOW_DAYS)
)

in_period = df["in_study_period"]
n_total = in_period.sum()
n_window = (in_period & df["ges_within_90d_before_dx"]).sum()

print(f"\nOf the {n_total:,} in-study-period K31.84 patients:")
print(f"  GES within 0-{WINDOW_DAYS} days before dx: {n_window:,} ({100 * n_window / n_total:.1f}%)")
print(f"  (for reference) GES any time before dx, no window: {(in_period & (ges_date < dx_date)).sum():,}")
print(f"  (for reference) GES before-or-same-day, no window:  {(in_period & (ges_date <= dx_date)).sum():,}")
