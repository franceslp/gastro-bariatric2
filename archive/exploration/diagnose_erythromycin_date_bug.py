"""
diagnose_erythromycin_date_bug.py

The original add_erythromycin_routes_to_cohort.py run correctly produced
first_erythromycin_date_after_dx / last_erythromycin_date_after_dx for tens
of thousands of patients (confirmed via the original run's printed output).
But today's master_cohort_K3184_diabetes_before_surgery.csv shows these two
columns as 100% NaN, even for patients who DO have erythromycin_routes_after_dx
populated.

This checks the ORIGINAL file directly (straight out of
add_erythromycin_routes_to_cohort.py, before any further merges) to isolate
whether the date values were lost in that script itself, or introduced by
a LATER script in the chain (add_bariatric_surgery_and_concurrency.py, or
today's build_master_cohort_file.py).
"""

import pandas as pd

ORIGINAL_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"
BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
MASTER_CSV = "master_cohort_K3184_diabetes_before_surgery.csv"

print("Checking the ORIGINAL file (straight out of add_erythromycin_routes_to_cohort.py)...")
orig = pd.read_csv(ORIGINAL_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "first_erythromycin_date_after_dx",
                             "last_erythromycin_date_after_dx", "erythromycin_routes_after_dx"])
n_orig_with_date = orig["first_erythromycin_date_after_dx"].notna().sum()
n_orig_with_route = orig["erythromycin_routes_after_dx"].notna().sum()
print(f"  Patients with a non-null first_erythromycin_date_after_dx: {n_orig_with_date:,}")
print(f"  Patients with a non-null erythromycin_routes_after_dx:     {n_orig_with_route:,}")

# THE KEY CHECK: does this exact "route present, date missing" pattern
# already exist in the ORIGINAL file, before anything downstream touches it?
# If yes, the bug is in add_erythromycin_routes_to_cohort.py's date-parsing
# logic itself (start_date failing to parse while route still does), not
# anything introduced later in the pipeline.
orig_problem = orig[orig["erythromycin_routes_after_dx"].notna() & orig["first_erythromycin_date_after_dx"].isna()]
print(f"  Patients with a route but NO date, already in the ORIGINAL file: {len(orig_problem):,}")
if len(orig_problem) > 0:
    print("  ^ BUG LOCATED: this already exists in add_erythromycin_routes_to_cohort.py's own output -")
    print("    not something introduced downstream. Likely a start_date parsing failure for some rows.")

print("\nChecking the BARIATRIC file (output of add_bariatric_surgery_and_concurrency.py)...")
bariatric = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False,
                         usecols=["patient_id", "first_erythromycin_date_after_dx",
                                  "last_erythromycin_date_after_dx", "erythromycin_routes_after_dx"])
n_bar_with_date = bariatric["first_erythromycin_date_after_dx"].notna().sum()
n_bar_with_route = bariatric["erythromycin_routes_after_dx"].notna().sum()
print(f"  Patients with a non-null first_erythromycin_date_after_dx: {n_bar_with_date:,}")
print(f"  Patients with a non-null erythromycin_routes_after_dx:     {n_bar_with_route:,}")

# If the bariatric file already lost the dates, this pinpoints the bug to
# add_bariatric_surgery_and_concurrency.py. If it still HAS the dates, the
# bug is introduced even later (today's merge script).
if n_orig_with_date > 0 and n_bar_with_date == 0:
    print("\n  ^ BUG LOCATED: dates are present in the original file but already gone")
    print("    by the time add_bariatric_surgery_and_concurrency.py wrote its output.")
elif n_bar_with_date > 0:
    print("\n  ^ Dates are still present in the bariatric file - bug must be in a later step.")

print("\nChecking today's MASTER file for comparison...")
master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False,
                      usecols=["patient_id", "first_erythromycin_date_after_dx", "erythromycin_routes_after_dx"])
n_master_with_date = master["first_erythromycin_date_after_dx"].notna().sum()
n_master_with_route = master["erythromycin_routes_after_dx"].notna().sum()
print(f"  Patients with a non-null first_erythromycin_date_after_dx: {n_master_with_date:,}")
print(f"  Patients with a non-null erythromycin_routes_after_dx:     {n_master_with_route:,}")

# Sanity check for a column-name collision during today's merges (would show
# up as _x/_y suffixed duplicates). Ruled out by code inspection already -
# neither e10e11_df nor rao_df's usecols include anything erythromycin-
# related - but confirming empirically costs nothing.
erythromycin_cols_in_master_source = [c for c in bariatric.columns if "erythromycin" in c]
print(f"\nErythromycin-related columns seen in the bariatric file: {erythromycin_cols_in_master_source}")
print("  (if any of these end in _x or _y, that indicates a merge collision - not expected here)")

# Spot-check: pick a few patients who HAD a date in the original file and see
# what happened to their row by the time it reached the bariatric file.
sample_ids = orig[orig["first_erythromycin_date_after_dx"].notna()]["patient_id"].head(5).tolist()
print(f"\nSpot-checking {len(sample_ids)} patients who had a date in the original file:")
for pid in sample_ids:
    orig_val = orig.loc[orig["patient_id"] == pid, "first_erythromycin_date_after_dx"].values
    bar_val = bariatric.loc[bariatric["patient_id"] == pid, "first_erythromycin_date_after_dx"].values
    orig_val = orig_val[0] if len(orig_val) > 0 else "NOT IN FILE"
    bar_val = bar_val[0] if len(bar_val) > 0 else "NOT IN FILE"
    print(f"  patient {pid}: original={orig_val!r}  ->  bariatric file={bar_val!r}")
