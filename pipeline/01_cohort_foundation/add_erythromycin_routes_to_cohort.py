"""
add_erythromycin_routes_to_cohort.py

Rolls up per-patient erythromycin route detail into the master cohort file,
so route info (oral/topical/ophthalmic/injectable/missing) is visible
directly on each patient's row, the same way diabetes type and GES detail
already are - rather than needing to separately filter
erythromycin_drug_detail_cohort_only.csv by patient_id.

Scoped to records ON/AFTER the patient's K31.84 diagnosis date, consistent
with how this has been reported throughout (the "any time" erythromycin
number includes pre-dx and unrelated antibiotic use, which isn't a
meaningful per-patient characteristic to merge in here).

A patient can have more than one route (e.g. an oral course AND a separate
ophthalmic prescription) - erythromycin_routes_after_dx lists every route
seen, comma-joined, and the individual boolean columns let you filter on
any specific route directly.

Run this AFTER add_diabetes_diagnosis_to_cohort.py. Reads two already-local
files (no GCS access needed), so this runs almost instantly. Writes a NEW
file rather than overwriting the diabetes output.
"""

import os
import pandas as pd

INPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_and_diabetes.csv"
DETAIL_CSV = "erythromycin_drug_detail_cohort_only.csv"
OUTPUT_CSV = "gastroparesis_prokinetic_cohort_with_GES_diabetes_and_erythromycin_routes.csv"

for path in (INPUT_CSV, DETAIL_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: {path}")

print(f"Loading {INPUT_CSV}...")
cohort_df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str})
cohort_df["patient_id"] = cohort_df["patient_id"].astype(str).str.strip()

print(f"Loading {DETAIL_CSV}...")
detail = pd.read_csv(DETAIL_CSV, dtype={"patient_id": str})
detail["patient_id"] = detail["patient_id"].astype(str).str.strip()

if "on_or_after_first_K31_84_dx" not in detail.columns:
    raise ValueError(f"{DETAIL_CSV} is missing the expected 'on_or_after_first_K31_84_dx' column")
if "route" not in detail.columns:
    raise ValueError(f"{DETAIL_CSV} is missing the expected 'route' column")
if "start_date" not in detail.columns:
    raise ValueError(f"{DETAIL_CSV} is missing the expected 'start_date' column")

detail["route"] = detail["route"].str.strip()
detail["start_date"] = pd.to_datetime(detail["start_date"], format="%Y%m%d", errors="coerce")
after_dx_mask = detail["on_or_after_first_K31_84_dx"].astype(str).str.lower() == "true"
after_dx = detail[after_dx_mask].copy()

ROUTE_CHECKS = {
    "oral": "oral",
    "topical": "topical",
    "ophthalmic": "ophthalmic",
    "injectable": "injectable",
}

# Per-patient rollup via groupby - one row per patient with every route seen.
routes_seen = {}
route_flags = {route: set() for route in ROUTE_CHECKS}
missing_route_patients = set()
first_erythromycin_date_after_dx = {}
last_erythromycin_date_after_dx = {}

for pid, sub in after_dx.groupby("patient_id"):
    distinct_routes = sub["route"].dropna().unique()
    if len(distinct_routes) > 0:
        routes_seen[pid] = ",".join(sorted(distinct_routes))
    if sub["route"].isna().any():
        missing_route_patients.add(pid)
    for route_name, keyword in ROUTE_CHECKS.items():
        if sub["route"].str.contains(keyword, case=False, na=False).any():
            route_flags[route_name].add(pid)
    min_date, max_date = sub["start_date"].min(), sub["start_date"].max()
    if pd.notna(min_date):
        first_erythromycin_date_after_dx[pid] = min_date
    if pd.notna(max_date):
        last_erythromycin_date_after_dx[pid] = max_date

cohort_df["erythromycin_routes_after_dx"] = cohort_df["patient_id"].map(routes_seen)
cohort_df["first_erythromycin_date_after_dx"] = cohort_df["patient_id"].map(first_erythromycin_date_after_dx)
cohort_df["last_erythromycin_date_after_dx"] = cohort_df["patient_id"].map(last_erythromycin_date_after_dx)
for route_name in ROUTE_CHECKS:
    cohort_df[f"erythromycin_{route_name}_after_dx"] = cohort_df["patient_id"].isin(route_flags[route_name])
cohort_df["erythromycin_missing_route_after_dx"] = cohort_df["patient_id"].isin(missing_route_patients)

print("\nSANITY CHECK:")
print(f"  cohort size: {len(cohort_df):,}")
for route_name in ROUTE_CHECKS:
    n = cohort_df[f"erythromycin_{route_name}_after_dx"].sum()
    print(f"  {route_name}: {n:,} patients ({100 * n / len(cohort_df):.1f}% of cohort)")
n_missing = cohort_df["erythromycin_missing_route_after_dx"].sum()
print(f"  missing route: {n_missing:,} patients ({100 * n_missing / len(cohort_df):.1f}% of cohort)")

cohort_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} with erythromycin_routes_after_dx (readable summary),")
print("first_erythromycin_date_after_dx, last_erythromycin_date_after_dx, plus")
print("erythromycin_oral_after_dx, erythromycin_topical_after_dx, erythromycin_ophthalmic_after_dx,")
print("erythromycin_injectable_after_dx, and erythromycin_missing_route_after_dx columns added.")
print(f"(Original {INPUT_CSV} left untouched - use the new file as input to the next step.)")
