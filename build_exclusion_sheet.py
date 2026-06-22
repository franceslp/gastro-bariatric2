"""
build_exclusion_sheet.py

Assembles all columns needed for the exclusion criteria sheet - a separate
tab in the master Excel file, NOT a modification of master_cohort_FINAL_1118.csv.

Exclusion criteria captured as flags (True = exclude, False = keep):
  1. exclude_K3184_too_far_before_surgery  (K31.84 > 1yr before surgery)
  2. exclude_E10E11_too_far_before_surgery (E10/E11 > 1yr before surgery)
  3. exclude_age_under_18                  (age < 18 at surgery)
  4. exclude_no_GES_within_1yr            (no GES within 1yr before K31.84)
  5. exclude_no_prokinetic_within_1yr     (no prokinetic within 1yr after K31.84)

Also includes:
  - The specific prokinetic drug used (within 1yr window)
  - Erythromycin route (oral/ophthalmic/topical etc.) where applicable
  - Whether prokinetic was prescribed before, same-day, or after surgery

No new scans - merges existing files only.
"""

import pandas as pd

OUTPUT_CSV = "exclusion_criteria_sheet.csv"
EXPECTED_ROWS = 1118

print(">>> SCRIPT VERSION: build_exclusion_sheet_v1 <<<")

master = pd.read_csv("master_cohort_FINAL_1118.csv", dtype={"patient_id": str},
                      low_memory=False,
                      usecols=["patient_id", "bariatric_date",
                               "closest_K31_84_strictly_before_surgery",
                               "meets_age_requirement", "age_at_surgery_approx",
                               "year_of_birth"])

print(f"Base: {len(master):,} patients")
df = master.copy()

k3184_flag = pd.read_csv("cohort_with_K3184_1yr_flag.csv", dtype={"patient_id": str},
                          low_memory=False,
                          usecols=["patient_id", "exclude_K3184_too_far_before_surgery",
                                   "days_K3184_before_surgery"])
df = df.merge(k3184_flag, on="patient_id", how="left")
assert len(df) == EXPECTED_ROWS, f"Row count changed after K31.84 flag merge: {len(df)}"

e10e11_flag = pd.read_csv("cohort_with_E10E11_1yr_flag.csv", dtype={"patient_id": str},
                           low_memory=False,
                           usecols=["patient_id", "exclude_E10E11_too_far_before_surgery",
                                    "days_E10E11_before_surgery"])
df = df.merge(e10e11_flag, on="patient_id", how="left")
assert len(df) == EXPECTED_ROWS, f"Row count changed after E10/E11 flag merge: {len(df)}"

df["exclude_age_under_18"] = (df["meets_age_requirement"] != True)  # != True treats NaN as exclude

ges_1yr = pd.read_csv("cohort_GES_before_closest_dx_1yr.csv", dtype={"patient_id": str},
                       low_memory=False,
                       usecols=["patient_id", "GES_before_closest_dx_1yr_date",
                                "GES_before_closest_dx_1yr_code",
                                "days_GES_before_K3184"])
df = df.merge(ges_1yr, on="patient_id", how="left")
assert len(df) == EXPECTED_ROWS, f"Row count changed after GES merge: {len(df)}"
df["exclude_no_GES_within_1yr"] = df["GES_before_closest_dx_1yr_date"].isna()

prok_1yr = pd.read_csv("cohort_prokinetic_with_surgery_timing_flags.csv",
                        dtype={"patient_id": str}, low_memory=False,
                        usecols=["patient_id", "prokinetic_after_closest_dx_1yr_date",
                                 "prokinetic_after_closest_dx_1yr_drug",
                                 "days_to_prokinetic_after_K3184",
                                 "prokinetic_before_surgery",
                                 "prokinetic_same_day_as_surgery",
                                 "prokinetic_after_surgery"])
df = df.merge(prok_1yr, on="patient_id", how="left")
assert len(df) == EXPECTED_ROWS, f"Row count changed after prokinetic merge: {len(df)}"
df["exclude_no_prokinetic_within_1yr"] = df["prokinetic_after_closest_dx_1yr_date"].isna()

# Erythromycin route - only for patients where erythromycin was the
# qualifying 1yr prokinetic, matched to the exact prokinetic date
ery_patients = set(df.loc[df["prokinetic_after_closest_dx_1yr_drug"] == "erythromycin", "patient_id"])
print(f"\nErythromycin patients (qualifying 1yr prokinetic): {len(ery_patients):,}")

ery_detail = pd.read_csv("erythromycin_drug_detail_cohort_only.csv",
                          dtype={"patient_id": str}, low_memory=False)
ery_detail = ery_detail[ery_detail["patient_id"].isin(ery_patients)].copy()
ery_detail["start_date_parsed"] = pd.to_datetime(ery_detail["start_date"], errors="coerce")

prok_date_lookup = dict(zip(
    df["patient_id"],
    pd.to_datetime(df["prokinetic_after_closest_dx_1yr_date"], errors="coerce")
))

erythromycin_route = {}
for pid in ery_patients:
    prok_date = prok_date_lookup.get(pid)
    if pd.isna(prok_date):
        continue
    patient_rows = ery_detail[
        (ery_detail["patient_id"] == pid) &
        ((ery_detail["start_date_parsed"] - prok_date).abs().dt.days <= 1)
    ]
    if not patient_rows.empty:
        routes = sorted(patient_rows["route"].dropna().unique())
        if routes:
            erythromycin_route[pid] = ", ".join(routes)

df["erythromycin_route"] = df["patient_id"].map(erythromycin_route)
n_with_route = df["erythromycin_route"].notna().sum()
print(f"Erythromycin patients with route identified: {n_with_route:,}/{len(ery_patients):,}")

def timing_label(row):
    if pd.isna(row["prokinetic_after_closest_dx_1yr_date"]):
        return None
    if row["prokinetic_before_surgery"]:
        return "before surgery"
    elif row["prokinetic_same_day_as_surgery"]:
        return "same day as surgery"
    elif row["prokinetic_after_surgery"]:
        return "after surgery"
    return None

df["prokinetic_timing_vs_surgery"] = df.apply(timing_label, axis=1)

# Safety cleanup: ensure all exclusion flags are proper booleans.
# Using map() rather than astype(bool) directly because CSV round-trips
# write True/False as the STRINGS "True"/"False", and bool("False") == True
# in Python - the map handles both native booleans and string forms safely.
flag_cols = [
    "exclude_K3184_too_far_before_surgery",
    "exclude_E10E11_too_far_before_surgery",
    "exclude_age_under_18",
    "exclude_no_GES_within_1yr",
    "exclude_no_prokinetic_within_1yr",
]

# Print BEFORE fill so missing rates are actually informative
# (after fill everything reads 0, which tells you nothing about coverage)
print("\nMissing merge coverage check (before fill):")
for c in flag_cols:
    missing_rate = df[c].isna().mean()
    n_missing = df[c].isna().sum()
    print(f"  {c}: {n_missing:,} missing ({missing_rate:.2%})")

BOOL_MAP = {True: True, False: False, "True": True, "False": False}
for c in flag_cols:
    df[c] = df[c].map(BOOL_MAP).fillna(False).astype(bool)

# CONSORT-style: how many exclusion reasons does each patient have?
df["number_exclusion_reasons"] = df[flag_cols].sum(axis=1)
print("\nDistribution of exclusion reason count per patient:")
print(df["number_exclusion_reasons"].value_counts().sort_index())

COLUMN_ORDER = [
    "patient_id",
    "year_of_birth",
    "age_at_surgery_approx",
    "meets_age_requirement",
    "exclude_age_under_18",
    "days_K3184_before_surgery",
    "exclude_K3184_too_far_before_surgery",
    "days_E10E11_before_surgery",
    "exclude_E10E11_too_far_before_surgery",
    "GES_before_closest_dx_1yr_date",
    "GES_before_closest_dx_1yr_code",
    "days_GES_before_K3184",
    "exclude_no_GES_within_1yr",
    "prokinetic_after_closest_dx_1yr_drug",
    "prokinetic_after_closest_dx_1yr_date",
    "days_to_prokinetic_after_K3184",
    "erythromycin_route",
    "prokinetic_timing_vs_surgery",
    "exclude_no_prokinetic_within_1yr",
    "number_exclusion_reasons",
]

df = df[COLUMN_ORDER]

print(f"\nExclusion summary (True = excluded by that criterion):")
for col in ["exclude_age_under_18", "exclude_K3184_too_far_before_surgery",
            "exclude_E10E11_too_far_before_surgery",
            "exclude_no_GES_within_1yr", "exclude_no_prokinetic_within_1yr"]:
    n = df[col].sum()
    print(f"  {col}: {n:,} ({100*n/len(df):.1f}%)")

all_excluded = (df["exclude_age_under_18"] |
                df["exclude_K3184_too_far_before_surgery"] |
                df["exclude_E10E11_too_far_before_surgery"] |
                df["exclude_no_GES_within_1yr"] |
                df["exclude_no_prokinetic_within_1yr"])
n_excluded = all_excluded.sum()
print(f"\nExcluded by ANY criterion: {n_excluded:,}/{len(df):,}")
print(f"Pass ALL criteria (final qualifying cohort): {len(df)-n_excluded:,}/{len(df):,}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(df):,} rows, {len(df.columns)} columns)")
print("(Upload as a second sheet in the master Excel file)")
