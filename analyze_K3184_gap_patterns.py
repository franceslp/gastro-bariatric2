"""
analyze_K3184_gap_patterns.py

For patients with multiple K31.84 dates, distinguishes coding patterns that
look identical under a single "first to last span" number:

  - persistent_regular_coding: no single gap between encounters exceeds a
    year - frequent administrative coding, NOT proof of continuous
    clinical disease activity
  - long_gap_pattern: exactly one gap exceeds a year, regardless of how
    many other (smaller) gaps surround it - could reflect a true
    recurrence (resolved, then came back) OR just infrequent coding of an
    otherwise ongoing condition - the data cannot distinguish these
  - intermittent_coding: multiple large gaps scattered throughout

These labels describe CODING PATTERN, not disease course - the data
genuinely cannot tell true recurrence apart from infrequently-coded
chronic disease. Avoid describing this as a "recurrence pattern" in any
write-up; "temporal coding persistence pattern" is more accurate.

No new scan - parses the already-collected all_K31_84_dates column from
cohort_K3184_full_history.csv.
"""

import pandas as pd
import statistics

INPUT_CSV = "cohort_K3184_full_history.csv"
OUTPUT_CSV = "cohort_K3184_gap_analysis.csv"

LARGE_GAP_DAYS = 365  # gaps >365 days flagged as large

df = pd.read_csv(INPUT_CSV, dtype={"patient_id": str}, low_memory=False)
print(f"Cohort size: {len(df):,} patients")


def parse_dates(date_str):
    if pd.isna(date_str):
        return []
    dates = []
    for d in date_str.split(","):
        try:
            dates.append(pd.to_datetime(d.strip()))
        except (ValueError, TypeError):
            continue
    # set() dedup is currently redundant (the upstream script that built this
    # column already collected dates into a Python set before writing them
    # out), but kept here as insurance in case this script is ever pointed
    # at a different, less-controlled input later.
    return sorted(set(dates))


def compute_gaps(dates):
    if len(dates) < 2:
        return []
    return [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]


max_gap_list = []
mean_gap_list = []
median_gap_list = []
num_large_gaps_list = []
num_encounters_list = []
span_list = []
density_list = []
pattern_label_list = []

for date_str in df["all_K31_84_dates"]:
    dates = parse_dates(date_str)
    gaps = compute_gaps(dates)
    num_encounters_list.append(len(dates))

    # Computed independently here, not reused from any other script's
    # output - makes this script self-contained rather than silently
    # depending on a column that could be missing or stale.
    if len(dates) >= 1:
        span_days = (dates[-1] - dates[0]).days
    else:
        span_days = pd.NA
    span_list.append(span_days)

    if not gaps:
        max_gap_list.append(pd.NA)
        mean_gap_list.append(pd.NA)
        median_gap_list.append(pd.NA)
        num_large_gaps_list.append(pd.NA)
        density_list.append(pd.NA)
        pattern_label_list.append("single_encounter")
        continue

    max_gap = max(gaps)
    mean_gap = sum(gaps) / len(gaps)
    median_gap = statistics.median(gaps)
    num_large = sum(1 for g in gaps if g > LARGE_GAP_DAYS)

    max_gap_list.append(max_gap)
    mean_gap_list.append(mean_gap)
    median_gap_list.append(median_gap)
    num_large_gaps_list.append(num_large)

    # Encounters per year - connects back to the actual motivating question
    # (is first K31.84 a reasonable anchor?) by normalizing encounter count
    # against how much time it's spread across, in one number.
    density = len(dates) / (span_days / 365) if span_days > 0 else pd.NA
    density_list.append(density)

    # Renamed and reclassified per review: avoids implying a CLINICAL claim
    # (continuous disease, true recurrence) when the data only supports an
    # ADMINISTRATIVE one (how often/how regularly this was coded). The data
    # genuinely cannot distinguish "chronic, intermittently coded" from
    # "resolved and later recurred" - these labels describe coding pattern,
    # not disease course.
    if max_gap <= LARGE_GAP_DAYS:
        pattern_label_list.append("persistent_regular_coding")
    elif num_large == 1:
        pattern_label_list.append("long_gap_pattern")
    else:
        pattern_label_list.append("intermittent_coding")

df["num_K31_84_encounters"] = num_encounters_list
df["K3184_span_days"] = span_list
df["max_gap_days"] = max_gap_list
df["mean_gap_days"] = mean_gap_list
df["median_gap_days"] = median_gap_list
df["num_gaps_over_1yr"] = num_large_gaps_list
df["encounters_per_year"] = density_list
df["K3184_pattern"] = pattern_label_list

print("\n" + "="*70)
print("PATTERN BREAKDOWN (all patients)")
print("="*70)
print(df["K3184_pattern"].value_counts())

n_multi = (df["num_K31_84_encounters"] > 1).sum()
print(f"\nMulti-encounter patients (the actual denominator for gap/density stats below): {n_multi:,}/{len(df):,}")

print("\n" + "="*70)
print("K31.84 SPAN DISTRIBUTION - directly informs the anchor-date question:")
print("is first_K31_84_date a reasonable proxy for disease onset?")
print("="*70)
print(df["K3184_span_days"].describe())

print("\n" + "="*70)
print("Of patients with a multi-year total span (>365 days first-to-last),")
print("what does their actual coding pattern look like?")
print("="*70)
# Uses the span computed independently above, not the external column from
# explore_K3184_full_history.py - this script no longer silently depends
# on that file's exact output format.
long_span = df["K3184_span_days"].notna() & (df["K3184_span_days"] > 365)
print(df[long_span]["K3184_pattern"].value_counts())
print(f"\n  -> 'persistent_regular_coding': spans over a year, but no single gap")
print(f"     between encounters ever exceeds a year - frequent administrative")
print(f"     coding, NOT proof of continuous clinical disease activity.")
print(f"  -> 'long_gap_pattern': exactly one gap over a year, regardless of how")
print(f"     many other encounters surround it - could reflect a true recurrence")
print(f"     OR just infrequent coding of an ongoing condition. The data cannot")
print(f"     distinguish these - this describes coding pattern, not disease course.")
print(f"  -> 'intermittent_coding': multiple large gaps scattered throughout.")

print("\n" + "="*70)
print("Encounter density (encounters per year of span) - among multi-encounter patients")
print("="*70)
print(df["encounters_per_year"].describe())

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
