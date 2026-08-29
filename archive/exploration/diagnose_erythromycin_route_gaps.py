"""
diagnose_erythromycin_route_gaps.py

9/14 erythromycin patients had a route identified. This checks the 5 that
didn't, comparing their qualifying prokinetic date (from the 1yr window scan)
against every erythromycin record in the detail file for that patient, to see
exactly why the match failed.

Possible causes:
  1. Date mismatch > 1 day (billing vs dispensing date lag)
  2. Patient not in the erythromycin detail file at all
  3. Route column is null for their matching records
  4. The qualifying prokinetic date itself is slightly different from
     what's in the detail file due to parsing differences
"""

import pandas as pd

PROK_CSV = "cohort_prokinetic_with_surgery_timing_flags.csv"
ERY_CSV = "erythromycin_drug_detail_cohort_only.csv"

print(">>> SCRIPT VERSION: diagnose_erythromycin_route_gaps_v1 <<<")

prok = pd.read_csv(PROK_CSV, dtype={"patient_id": str}, low_memory=False,
                    usecols=["patient_id", "prokinetic_after_closest_dx_1yr_drug",
                             "prokinetic_after_closest_dx_1yr_date"])

ery_patients_all = set(prok.loc[prok["prokinetic_after_closest_dx_1yr_drug"] == "erythromycin", "patient_id"])
print(f"Total erythromycin qualifying patients: {len(ery_patients_all):,}")

ery = pd.read_csv(ERY_CSV, dtype={"patient_id": str}, low_memory=False)
ery["start_date_parsed"] = pd.to_datetime(ery["start_date"], errors="coerce")

prok_date_lookup = dict(zip(
    prok["patient_id"],
    pd.to_datetime(prok["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
))

matched = set()
for pid in ery_patients_all:
    prok_date = prok_date_lookup.get(pid)
    if pd.isna(prok_date):
        continue
    patient_rows = ery[(ery["patient_id"] == pid) &
                       ((ery["start_date_parsed"] - prok_date).abs().dt.days <= 1)]
    if not patient_rows.empty:
        routes = patient_rows["route"].dropna().unique()
        if len(routes) > 0:
            matched.add(pid)

unmatched = ery_patients_all - matched
print(f"Matched: {len(matched):,}, Unmatched: {len(unmatched):,}")

print("\n" + "="*70)
print("DETAILED DIAGNOSIS FOR EACH UNMATCHED PATIENT:")
print("="*70)

for pid in sorted(unmatched):
    prok_date = prok_date_lookup.get(pid)
    print(f"\nPatient: {pid}")
    print(f"  Qualifying prokinetic date: {prok_date}")

    all_ery_rows = ery[ery["patient_id"] == pid]
    if all_ery_rows.empty:
        print("  -> NOT IN erythromycin detail file at all")
        continue

    print(f"  All erythromycin records in detail file ({len(all_ery_rows):,} rows):")
    for _, row in all_ery_rows.iterrows():
        gap = None
        if pd.notna(row["start_date_parsed"]) and pd.notna(prok_date):
            gap = abs((row["start_date_parsed"] - prok_date).days)
        print(f"    start_date={row['start_date']} parsed={row['start_date_parsed']} "
              f"route={row['route']} gap_days={gap}")
