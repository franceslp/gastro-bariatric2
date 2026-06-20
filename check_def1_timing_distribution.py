"""
check_def1_timing_distribution.py

For the 338 patients meeting Definition 1 (prokinetic after dx + GES
before/on dx) + bariatric surgery, what's the actual TIMING distribution?

GES timing: no new scan needed - first_GES_date and first_K31_84_date are
already in the master file, this is pure arithmetic.

Prokinetic timing: NEEDS a new scan. The any_prokinetic_ever_after_dx flag
used to build Definition 1 is just a yes/no boolean - no date was ever
saved for WHEN the prokinetic was first used after diagnosis. This scans
medication_ingredient.csv, restricted to just these 338 patients, to find
each patient's earliest prokinetic record (any of the 4 drugs) dated on or
after their K31.84 diagnosis.
"""

import subprocess
import time
import pandas as pd
from pandas.api.types import is_bool_dtype

print(">>> SCRIPT VERSION: def1_timing_distribution_v1 <<<")

SCRIPT_START_TIME = time.time()

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MED_INGREDIENT_FILE = f"{GCS_BASE}/medication_ingredient.csv"

BARIATRIC_CSV = "gastroparesis_prokinetic_cohort_FULL_with_bariatric_and_concurrency.csv"
OUTPUT_CSV = "def1_bariatric_patients_with_timing.csv"

PROKINETIC_RXNORM_CODES = {
    "6915": "metoclopramide",
    "4053": "erythromycin",
    "3626": "domperidone",
    "2107310": "prucalopride",
}

df = pd.read_csv(BARIATRIC_CSV, dtype={"patient_id": str}, low_memory=False)

if not is_bool_dtype(df["has_bariatric_surgery"]):
    df["has_bariatric_surgery"] = df["has_bariatric_surgery"].astype(str).str.strip().str.lower().eq("true")

ges_dt = pd.to_datetime(df["first_GES_date"], errors="coerce")
dx_dt = pd.to_datetime(df["first_K31_84_date"], errors="coerce")
ges_before_or_same_day = ges_dt.notna() & dx_dt.notna() & (ges_dt <= dx_dt)
def1_met = (
    df["first_K31_84_date"].notna()
    & df["any_prokinetic_ever_after_dx"].fillna(False)
    & ges_before_or_same_day
)

mask = def1_met & df["in_study_period"] & df["has_bariatric_surgery"]
cohort = df[mask].copy()
cohort_ids = set(cohort["patient_id"])
print(f"Definition 1 + bariatric surgery population: {len(cohort_ids):,} patients\n")

# --- GES timing: arithmetic only, no scan ---
cohort_ges_dt = pd.to_datetime(cohort["first_GES_date"], errors="coerce")
cohort_dx_dt = pd.to_datetime(cohort["first_K31_84_date"], errors="coerce")
# 0 = same-day GES (matches the eligibility test above, ges_dt <= dx_dt,
# which allows same-day); negative would mean GES after dx, which shouldn't
# occur here and is explicitly checked further down (n_ges_after_dx).
days_GES_before_dx = (cohort_dx_dt - cohort_ges_dt).dt.days
cohort["days_GES_before_dx"] = days_GES_before_dx

print("\n" + "="*70)
print("PART 1: DESCRIPTIVE TIMING (among Definition 1 + bariatric patients)")
print("="*70)
print("GES TIMING (days before K31.84 diagnosis):")
print(f"  mean:   {days_GES_before_dx.mean():,.0f} days (~{days_GES_before_dx.mean()/365.25:.1f} years)")
print(f"  median: {days_GES_before_dx.median():,.0f} days (~{days_GES_before_dx.median()/365.25:.1f} years)")
print(f"  min:    {days_GES_before_dx.min():,.0f} days")
print(f"  max:    {days_GES_before_dx.max():,.0f} days (~{days_GES_before_dx.max()/365.25:.1f} years)")
# Mean/median alone understate skew - both these distributions are visibly
# right-skewed (max stretches to years while median is much smaller), so
# percentiles give a clearer picture of the actual shape.
ges_pctiles = days_GES_before_dx.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print(f"  percentiles (10/25/50/75/90): {ges_pctiles.round(0).astype(int).tolist()}")

bins = [0, 90, 365, 730, 1825, 100000]
labels = ["0-90 days", "91-365 days", "1-2 years", "2-5 years", "5+ years"]
print("\nBreakdown:")
print(pd.cut(days_GES_before_dx, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index())

# Extra granularity for comparing against Rao's GES-timing piece specifically
# (one of Rao's three requirements - symptoms and endoscopy are separate and
# not checked here) - helps show how much of Definition 1's population would
# also satisfy that one piece of Rao's logic.
print(f"\nGES same-day as dx:    {(days_GES_before_dx == 0).sum():,}")
print(f"GES within 90 days:    {(days_GES_before_dx <= 90).sum():,}")

# --- Prokinetic timing: needs a fresh scan, restricted to this small group ---
dx_date_lookup = dict(zip(cohort["patient_id"], cohort_dx_dt))

print(f"\nScanning medication_ingredient.csv for prokinetic codes, restricted to {len(cohort_ids):,} patients...")

# Tracked PER DRUG (not just an overall min) so we can report which specific
# drug was the first one used after dx for each patient - useful for the
# manuscript question "were these mostly metoclopramide-treated patients?"
first_after_dx_by_drug = {name: {} for name in PROKINETIC_RXNORM_CODES.values()}
rows_seen = 0
chunk_num = 0


def stream_gcs_csv(gcs_path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", gcs_path], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close()
    proc.wait()


for chunk in stream_gcs_csv(MED_INGREDIENT_FILE, usecols=["patient_id", "code_system", "code", "start_date"]):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(cohort_ids)]
    if not chunk.empty:
        rx = chunk[(chunk["code_system"] == "RxNorm") & (chunk["code"].isin(PROKINETIC_RXNORM_CODES.keys()))].copy()
        if not rx.empty:
            rx["start_date"] = pd.to_datetime(rx["start_date"], format="%Y%m%d", errors="coerce")
            rx["dx_date"] = rx["patient_id"].map(dx_date_lookup)
            after_dx = rx[rx["dx_date"].notna() & rx["start_date"].notna() & (rx["start_date"] >= rx["dx_date"])].copy()
            if not after_dx.empty:
                after_dx["drug"] = after_dx["code"].map(PROKINETIC_RXNORM_CODES)
                # vectorized lookup instead of looping over each of the 4 codes
                # separately and re-filtering the same chunk 4 times
                for (pid, name), d in after_dx.groupby(["patient_id", "drug"])["start_date"].min().items():
                    if pid not in first_after_dx_by_drug[name] or d < first_after_dx_by_drug[name][pid]:
                        first_after_dx_by_drug[name][pid] = d

    if chunk_num % 50 == 0:
        elapsed = (time.time() - SCRIPT_START_TIME) / 60
        print(f"    ...{rows_seen:,} rows scanned ({elapsed:.1f} min elapsed)")

print(f"  done - scanned {rows_seen:,} rows\n")

# Collapse the per-drug dicts down to: each patient's single earliest
# after-dx prokinetic date AND which drug that was.
first_prokinetic_after_dx = {}
first_prokinetic_name_after_dx = {}
for name, pid_dates in first_after_dx_by_drug.items():
    for pid, d in pid_dates.items():
        if pid not in first_prokinetic_after_dx or d < first_prokinetic_after_dx[pid]:
            first_prokinetic_after_dx[pid] = d
            first_prokinetic_name_after_dx[pid] = name

cohort["first_prokinetic_date_after_dx"] = cohort["patient_id"].map(first_prokinetic_after_dx)
cohort["first_prokinetic_date_after_dx"] = pd.to_datetime(cohort["first_prokinetic_date_after_dx"], errors="coerce")
cohort["first_prokinetic_name_after_dx"] = cohort["patient_id"].map(first_prokinetic_name_after_dx)
days_prokinetic_after_dx = (cohort["first_prokinetic_date_after_dx"] - cohort_dx_dt).dt.days
cohort["days_prokinetic_after_dx"] = days_prokinetic_after_dx

# Tie transparency: if two+ drugs share a patient's exact earliest after-dx
# date, first_prokinetic_name_after_dx above just picks whichever drug came
# first in PROKINETIC_RXNORM_CODES's dict order (metoclopramide first) - not
# a meaningful tiebreak. This counts how often that actually happens, so the
# "which drug was first" breakdown below can be read with that caveat in
# mind rather than assumed to be exact for every patient.
n_same_day_ties = 0
tie_combinations = {}
for pid, win_date in first_prokinetic_after_dx.items():
    drugs_on_that_date = sorted(
        name for name in PROKINETIC_RXNORM_CODES.values()
        if first_after_dx_by_drug[name].get(pid) == win_date
    )
    if len(drugs_on_that_date) > 1:
        n_same_day_ties += 1
        combo = " + ".join(drugs_on_that_date)
        tie_combinations[combo] = tie_combinations.get(combo, 0) + 1
print(f"Patients with a same-day tie between 2+ prokinetics (tiebreak is arbitrary): {n_same_day_ties:,}/{len(cohort):,}")
if tie_combinations:
    print("  Tie combinations observed:")
    for combo, n in sorted(tie_combinations.items(), key=lambda x: -x[1]):
        print(f"    {combo}: {n:,}")
# Suggested methods-section language for this choice, if it comes up:
# "First prokinetic agent was determined by earliest recorded exposure;
#  patients with multiple same-day prokinetic exposures were classified by
#  a fixed priority order (metoclopramide > erythromycin > domperidone >
#  prucalopride) rather than treated as having multiple initial agents."

# Sanity check: every patient in this cohort passed Definition 1, which
# requires any_prokinetic_ever_after_dx == True - so this scan should find
# a qualifying date for ALL of them. If not, the RxNorm codes used here
# don't fully reproduce that original flag, and the timing numbers below
# are incomplete (missing the patients this scan couldn't match).
missing_prokinetic_dates = cohort["first_prokinetic_date_after_dx"].isna().sum()
missing_pct = 100 * missing_prokinetic_dates / len(cohort) if len(cohort) > 0 else 0
print(f"Patients meeting Definition 1 but missing a scanned prokinetic date: {missing_prokinetic_dates:,}/{len(cohort):,} ({missing_pct:.1f}%)")
if missing_prokinetic_dates > 0:
    print("  ^ NOT ZERO - possible causes include: RxNorm code-list mismatch, granularity")
    print("    differences (ingredient vs. clinical-drug concepts), or inpatient/outpatient")
    print("    encoding differences between this scan and the original cohort build.")
    print("    A handful (~1-2%) could just be edge cases; a large share warrants investigation.")
    print("    Timing stats below are based only on the patients successfully matched.")

# Sanity check: Definition 1 also requires GES before/on dx - first_GES_date
# should never be missing for anyone in this cohort.
missing_GES = cohort["first_GES_date"].isna().sum()
print(f"Patients meeting Definition 1 but missing a GES date: {missing_GES:,}/{len(cohort):,} (should be 0)")

matched = days_prokinetic_after_dx.notna()
print(f"PROKINETIC TIMING (among {matched.sum():,}/{len(cohort):,} patients with a matched prokinetic date):")
# Scope note: this is conditional exposure timing among patients who were
# ALREADY required to have prokinetic use after dx to qualify for Definition
# 1 in the first place. It describes "time to treatment among the treated,"
# not the probability of being treated at all or the general disease course.
print(f"  mean:   {days_prokinetic_after_dx.mean():,.0f} days (~{days_prokinetic_after_dx.mean()/365.25:.1f} years)")
print(f"  median: {days_prokinetic_after_dx.median():,.0f} days (~{days_prokinetic_after_dx.median()/365.25:.1f} years)")
print(f"  min:    {days_prokinetic_after_dx.min():,.0f} days")
print(f"  max:    {days_prokinetic_after_dx.max():,.0f} days (~{days_prokinetic_after_dx.max()/365.25:.1f} years)")
prokinetic_pctiles = days_prokinetic_after_dx.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print(f"  percentiles (10/25/50/75/90): {prokinetic_pctiles.round(0).astype(int).tolist()}")

# Sensitivity check: the primary metric above uses >= dx_date (matching the
# upstream any_prokinetic_ever_after_dx flag's own convention exactly, so
# Definition 1 membership and this timing analysis stay consistent). This
# full side-by-side comparison shows the same stats computed under a strict
# > dx_date definition (excluding day-0/same-day exposure, which could
# reflect inpatient or discharge prescribing rather than delayed outpatient
# initiation) - not just the median, the same mean/percentile detail as the
# primary metric above.
strict_after_dx = days_prokinetic_after_dx[days_prokinetic_after_dx > 0]
n_day_zero = (days_prokinetic_after_dx == 0).sum()
print(f"\n  Of which day-0 (same-day as dx) prokinetic exposure: {n_day_zero:,} ({100*n_day_zero/matched.sum():.1f}% of matched patients)")
print(f"  SENSITIVITY (strict > dx_date, excluding day-0, n={len(strict_after_dx):,}):")
print(f"    mean:   {strict_after_dx.mean():,.0f} days")
print(f"    median: {strict_after_dx.median():,.0f} days")
strict_pctiles = strict_after_dx.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
print(f"    percentiles (10/25/50/75/90): {strict_pctiles.round(0).astype(int).tolist()}")

print("\nBreakdown:")
print(pd.cut(days_prokinetic_after_dx, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index())

print("\nWhich drug was the FIRST prokinetic used after dx:")
print(cohort["first_prokinetic_name_after_dx"].value_counts())

# Defensive checks: Definition 1 requires GES <= dx and prokinetic >= dx, so
# neither of these should ever be negative. A nonzero count here would mean
# a hidden date-parsing problem, not a real data pattern.
n_ges_after_dx = (days_GES_before_dx < 0).sum()
n_prokinetic_before_dx = (days_prokinetic_after_dx < 0).sum()
print(f"\nGES dated AFTER dx (should be 0): {n_ges_after_dx:,}")
print(f"Prokinetic dated BEFORE dx (should be 0): {n_prokinetic_before_dx:,}")

print("\n" + "="*70)
print("PART 2: RAO-LIKE ELIGIBILITY OVERLAP (binary rule comparison, separate")
print("        question from Part 1's descriptive timing above)")
print("="*70)
# Directly answers the question this whole comparison is really about: is
# Definition 1 just a looser version of Rao, or does a meaningful chunk of
# it ALSO satisfy a Rao-like tight-timing pattern on both ends (test close
# to dx, treatment started close to dx)? This is a constructed proximity
# heuristic, NOT a formal subset of Rao's criteria - Rao also requires
# symptoms + endoscopy, which this doesn't check.
rao_timing_component_only = (
    days_GES_before_dx.notna()
    & days_prokinetic_after_dx.notna()
    & (days_GES_before_dx <= 90)
    & (days_prokinetic_after_dx <= 90)
)
n_rao_timing_component = rao_timing_component_only.sum()
# Denominator is deliberately len(cohort), not matched.sum(): a patient
# with no matched prokinetic date has no confirmed evidence of tight timing,
# so they correctly count as NOT meeting this criterion (the .notna() check
# above already excludes them from the numerator). Using the full cohort as
# the denominator is the conservative, correct framing here - switching to
# matched.sum() would artificially inflate this percentage by shrinking the
# denominator only on rows that already failed to show evidence.
print(f"\nGES within 90 days AND prokinetic within 90 days after dx (Rao-like timing overlap heuristic, not full Rao criteria): "
      f"{n_rao_timing_component:,}/{len(cohort):,} ({100*n_rao_timing_component/len(cohort):.1f}% of full Definition-1 cohort)")
if matched.sum() > 0:
    print(f"  (same numerator, alternate denominator: {n_rao_timing_component:,}/{matched.sum():,} = "
          f"{100*n_rao_timing_component/matched.sum():.1f}% of patients with a matched prokinetic date)")

# Which side is driving the patients that DON'T meet that tight timing -
# loose GES timing, loose prokinetic timing, or both? Directly answers the
# PI question: are we losing patients to Rao's stricter criteria because of
# the diagnostic confirmation (GES) timing, the treatment (prokinetic)
# timing, or both?
n_ges_loose = (days_GES_before_dx > 90).sum()
n_prokinetic_loose = (days_prokinetic_after_dx > 90).sum()
print(f"  GES >90 days before dx:           {n_ges_loose:,} ({100*n_ges_loose/len(cohort):.1f}%)")
print(f"  Prokinetic >90 days after dx:     {n_prokinetic_loose:,} ({100*n_prokinetic_loose/len(cohort):.1f}%)")

only_GES_loose = (days_GES_before_dx > 90) & (days_prokinetic_after_dx <= 90)
only_prokinetic_loose = (days_GES_before_dx <= 90) & (days_prokinetic_after_dx > 90)
both_loose = (days_GES_before_dx > 90) & (days_prokinetic_after_dx > 90)

n_only_ges = only_GES_loose.sum()
n_only_prokinetic = only_prokinetic_loose.sum()
n_both = both_loose.sum()
print(f"\nOf the patients failing tight (90-day) timing, the REASON breaks down as:")
print(f"  GES timing is the reason (prokinetic was tight):       {n_only_ges:,} ({100*n_only_ges/len(cohort):.1f}%)")
print(f"  Treatment timing is the reason (GES was tight):        {n_only_prokinetic:,} ({100*n_only_prokinetic/len(cohort):.1f}%)")
print(f"  Both GES and treatment timing are loose:                {n_both:,} ({100*n_both/len(cohort):.1f}%)")

cohort.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
print(f"Total runtime: {(time.time()-SCRIPT_START_TIME)/60:.1f} minutes")
