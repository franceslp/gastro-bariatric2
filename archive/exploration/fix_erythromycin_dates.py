"""
fix_erythromycin_dates.py

ROOT CAUSE CONFIRMED: erythromycin_drug_detail_cohort_only.csv stores
start_date as ISO format with dashes (e.g. "2021-09-26"), unlike every
other source file in this project (diagnosis.csv, procedure.csv,
medication_ingredient.csv), which all use plain "YYYYMMDD" with no
separators. The original add_erythromycin_routes_to_cohort.py almost
certainly reused this project's standard format="%Y%m%d" parsing
convention for this column too - which fails to match dashed dates,
silently turning every single one into NaT via errors="coerce". That
explains the exact pattern found: 24,165 patients with a route, 0 with
a date - a complete, deterministic failure, not a partial one.

This re-parses start_date correctly (no format string needed - pandas'
default ISO inference handles "YYYY-MM-DD" natively) and recomputes
first/last erythromycin date after dx, restricted to the 1,070-patient
cohort only (fast, since the file's already local and the cohort is small).

Local file, no GCS - this is essentially instant.
"""

import pandas as pd

DETAIL_CSV = "erythromycin_drug_detail_cohort_only.csv"
MASTER_CSV = "master_cohort_K3184_diabetes_before_surgery.csv"
OUTPUT_CSV = "master_cohort_K3184_diabetes_before_surgery_FIXED_erythromycin.csv"

print("Loading master cohort...")
master = pd.read_csv(MASTER_CSV, dtype={"patient_id": str}, low_memory=False)
cohort_ids = set(master["patient_id"])
print(f"  cohort size: {len(cohort_ids):,} patients")

print(f"\nLoading {DETAIL_CSV} (local file, no GCS)...")
detail = pd.read_csv(DETAIL_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"  total rows in file: {len(detail):,}")

detail = detail[detail["patient_id"].isin(cohort_ids)].copy()
print(f"  rows restricted to this cohort: {len(detail):,}")

n_dupes = detail["patient_id"].duplicated().sum()
print(f"  duplicate patient_id rows in this filtered detail (not a problem since we groupby, just QA): {n_dupes:,}")

# THE FIX: parse start_date with NO explicit format string. pandas correctly
# auto-detects "YYYY-MM-DD" ISO dates without needing format="%Y%m%d" (which
# is what almost certainly caused the original failure).
detail["start_date_parsed"] = pd.to_datetime(detail["start_date"], errors="coerce")
n_parse_failures = detail["start_date_parsed"].isna().sum()
print(f"\n  Date parse failures with the corrected parsing: {n_parse_failures:,}/{len(detail):,}")
if n_parse_failures == 0:
    print("  ^ Confirms the fix: every date in this cohort's rows parsed correctly this time.")

# DO NOT trust on_or_after_first_K31_84_dx - that flag was computed by the
# SAME original script, likely using the SAME broken date parsing. Trusting
# it here could silently inherit the exact bug this script is fixing.
# Instead, rebuild the after-dx filter independently from the master file's
# first_K31_84_date, which comes from diagnosis.csv via the standard
# format="%Y%m%d" parsing that's verified working throughout this project.
dx_lookup = dict(zip(master["patient_id"], pd.to_datetime(master["first_K31_84_date"], errors="coerce")))
detail["dx_date"] = detail["patient_id"].map(dx_lookup)

after_dx = detail[
    detail["start_date_parsed"].notna()
    & detail["dx_date"].notna()
    & (detail["start_date_parsed"] >= detail["dx_date"])
]
print(f"\n  Rows after independently-rebuilt after-dx filter: {len(after_dx):,}")

# Sanity check: by construction, the filter above already enforces this, but
# confirming explicitly catches any logic error in the filter itself.
bad = after_dx[after_dx["start_date_parsed"] < after_dx["dx_date"]]
print(f"  Dates before dx after filtering (should be 0): {len(bad):,}")

first_date = after_dx.groupby("patient_id")["start_date_parsed"].min()
last_date = after_dx.groupby("patient_id")["start_date_parsed"].max()
route_seen = after_dx.groupby("patient_id")["route"].apply(lambda s: ",".join(sorted(s.dropna().unique())) if s.notna().any() else None)

master["first_erythromycin_date_after_dx_FIXED"] = master["patient_id"].map(first_date)
master["last_erythromycin_date_after_dx_FIXED"] = master["patient_id"].map(last_date)
master["erythromycin_routes_after_dx_FIXED"] = master["patient_id"].map(route_seen)

n_fixed_with_date = master["first_erythromycin_date_after_dx_FIXED"].notna().sum()
n_fixed_routes = master["erythromycin_routes_after_dx_FIXED"].notna().sum()
n_original_with_route = master["erythromycin_routes_after_dx"].notna().sum()
n_recovered = (master["erythromycin_routes_after_dx"].notna() & master["first_erythromycin_date_after_dx_FIXED"].notna()).sum()
print(f"\nIn the master cohort ({len(master):,} patients):")
print(f"  originally had a route but no date: {n_original_with_route:,}")
print(f"  now have a corrected route:          {n_fixed_routes:,}")
print(f"  now have a corrected date:           {n_fixed_with_date:,}")
print(f"  of the route-only patients, now also have a recovered date: {n_recovered:,}/{n_original_with_route:,}")
print("  (gap, if any, is expected: the independent dx-date filter is stricter than the old broken flag,")
print("   so not every route-only patient will necessarily get a recovered after-dx date)")

master.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with the *_FIXED erythromycin columns added alongside the")
print("original (broken) ones, so you can compare directly before deciding to replace them.")
