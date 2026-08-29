"""
filter_erythromycin_detail_to_cohort.py

erythromycin_drug_detail.csv currently contains erythromycin records for
EVERY patient in the full dataset who ever had erythromycin (2,098,762
records), not just gastroparesis patients. This filters it down to only
patients who appear in gastroparesis_prokinetic_cohort.csv, which is what
you actually want to review.

IMPORTANT distinction: "any erythromycin record" includes records from
before the K31.84 diagnosis, antibiotic courses, and topical/ophthalmic use
- none of which suggest gastroparesis management. The detail file already
carries an on_or_after_first_K31_84_dx flag per record (computed in the
main build script), so this script reports BOTH numbers separately rather
than collapsing them into one count that could be misread as "treated for
gastroparesis with erythromycin."

Cheap and fast - reads two already-existing local CSVs, no GCS access.
"""

import os
import pandas as pd

for path in ("gastroparesis_prokinetic_cohort.csv", "erythromycin_drug_detail.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: {path}")

cohort = pd.read_csv("gastroparesis_prokinetic_cohort.csv", dtype={"patient_id": str})
detail = pd.read_csv("erythromycin_drug_detail.csv", dtype={"patient_id": str})

if "route" not in detail.columns:
    raise ValueError("erythromycin_drug_detail.csv is missing the expected 'route' column")
if "on_or_after_first_K31_84_dx" not in detail.columns:
    raise ValueError("erythromycin_drug_detail.csv is missing the expected 'on_or_after_first_K31_84_dx' column")

# Normalize patient_id formatting in case of stray whitespace - cheap insurance
# against a silent mismatch between the two files.
cohort["patient_id"] = cohort["patient_id"].astype(str).str.strip()
detail["patient_id"] = detail["patient_id"].astype(str).str.strip()

cohort_ids = set(cohort["patient_id"])
detail_filtered = detail[detail["patient_id"].isin(cohort_ids)].copy()
# Strip whitespace from route without converting NaN to the literal string
# "nan" - .str.strip() on its own correctly leaves real NaN values as NaN,
# unlike .astype(str).str.strip() which would corrupt the isna() check below.
detail_filtered["route"] = detail_filtered["route"].str.strip()
detail_filtered.to_csv("erythromycin_drug_detail_cohort_only.csv", index=False)

print(f"on_or_after_first_K31_84_dx dtype as read from CSV: {detail_filtered['on_or_after_first_K31_84_dx'].dtype}")
after_dx_mask = detail_filtered["on_or_after_first_K31_84_dx"].astype(str).str.lower() == "true"
after_dx = detail_filtered[after_dx_mask]
after_dx_oral = after_dx[after_dx["route"].str.contains("oral", case=False, na=False)]
after_dx_topical = after_dx[after_dx["route"].str.contains("topical", case=False, na=False)]
after_dx_ophthalmic = after_dx[after_dx["route"].str.contains("ophthalmic", case=False, na=False)]
# Added after the first real run surfaced this in the route value_counts -
# IV/injectable erythromycin is a recognized off-label prokinetic agent in
# hospitalized patients, arguably MORE specific to gastroparesis management
# than oral use (oral erythromycin's far more common indication is routine
# antibiotic treatment).
after_dx_injectable = after_dx[after_dx["route"].str.contains("injectable", case=False, na=False)]

cohort_n = len(cohort_ids)  # matches the exact set used for filtering above, not a separate nunique() call


def pct(n):
    return f"{100 * n / cohort_n:.1f}%" if cohort_n else "n/a"


any_n = detail_filtered["patient_id"].nunique()
after_dx_n = after_dx["patient_id"].nunique()
oral_n = after_dx_oral["patient_id"].nunique()
topical_n = after_dx_topical["patient_id"].nunique()
ophthalmic_n = after_dx_ophthalmic["patient_id"].nunique()
injectable_n = after_dx_injectable["patient_id"].nunique()
unknown_route_n = after_dx[after_dx["route"].isna()]["patient_id"].nunique()

print(f"Filtered from {len(detail):,} to {len(detail_filtered):,} rows")
print(f"Patients in cohort: {cohort_n:,}")
print(f"\nCohort patients with ANY erythromycin record (any time, any route, any reason):")
print(f"  {any_n:,} ({pct(any_n)} of cohort)")
print(f"  -- interpret this as 'any erythromycin exposure', NOT 'treated for gastroparesis'")
print(f"\nCohort patients with an erythromycin record specifically ON/AFTER their K31.84 dx:")
print(f"  {after_dx_n:,} ({pct(after_dx_n)} of cohort)")
print(f"  -- closer to a meaningful signal, though still can't rule out antibiotic use")
print(f"\nOf those on/after-dx patients, breakdown by route (% is of full cohort, not just after-dx group):")
print(f"  oral:       {oral_n:,} ({pct(oral_n)})  (rules out topical/ophthalmic)")
print(f"  topical:    {topical_n:,} ({pct(topical_n)})  (acne treatment, not gastroparesis)")
print(f"  ophthalmic: {ophthalmic_n:,} ({pct(ophthalmic_n)})  (eye infection, not gastroparesis)")
print(f"  injectable: {injectable_n:,} ({pct(injectable_n)})  (IV erythromycin is a recognized off-label")
print("              prokinetic agent in hospitalized patients - arguably MORE specific to")
print("              gastroparesis management than oral use, worth flagging to Dr. Sujka)")
print(f"  missing route (NaN, likely a failed medication_drug.csv join): {unknown_route_n:,} ({pct(unknown_route_n)})")
print("  -- note: this is distinct from the literal string 'Unknown' TriNetX uses when an HCO")
print("     didn't specify a route - that shows up as its own row in the value_counts below.")
print("     Missing route here means the oral/topical/ophthalmic numbers can't see these patients")
print("     at all, so a high count here is a reason to treat the oral number as a lower bound.")
print("  -- NaN routes are excluded from ALL THREE route categories above by design (str.contains")
print("     with na=False treats them as non-matching), not double-counted or silently dropped.")
print("\nNote: a patient with erythromycin via more than one route (e.g. an oral course AND")
print("a separate ophthalmic prescription) counts in both categories above, so oral + topical")
print("+ ophthalmic can add up to more than the on/after-dx total. That's expected, not an error.")
route_category_sum = oral_n + topical_n + ophthalmic_n + injectable_n
print(f"Total after-dx patients: {after_dx_n:,}")
print(f"Sum of route categories (not deduplicated, will not equal the total above): {route_category_sum:,}")
print(f"\nAll other route values present in the on/after-dx group (for full visibility):")
print(after_dx["route"].value_counts(dropna=False))
