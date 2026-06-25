#!/usr/bin/env python3
"""
step5_multisurgery_exclusion.py

Final exclusions (both confirmed by Dr. Sujka, June 2026) applied to the funnel
output, producing the final analytic cohort.

  Exclude multi-surgery       (_multi_surgery_flag): bariatric codes on >1 distinct
                              date (includes years-apart revision/conversion). Rationale:
                              avoid contaminating a procedure group with patients who
                              later underwent a different bariatric procedure.
  Exclude same-day ambiguous  (_same_day_ambiguous): >1 distinct bariatric CPT on the
                              SAME date -- i.e. same-day conflicting bariatric CPT coding
                              (could be conversion, duplicate billing, or coding artifact;
                              we exclude rather than adjudicate intent).

Input:  funnel_step4_ges.csv
Output: cohort_FINAL_analytic.csv
        excluded_multisurgery_sameday.csv

Usage:  python3 step5_multisurgery_exclusion.py
"""

import pandas as pd

INPUT_CSV = "funnel_step4_ges.csv"
FINAL_CSV = "cohort_FINAL_analytic.csv"
EXCLUDED_CSV = "excluded_multisurgery_sameday.csv"


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


def categorize(row):
    """Neutral description of why a patient was excluded, for the methods write-up."""
    sd = str(row.get("_same_day_ambiguous", "")).lower() in ("true", "1")
    ms = str(row.get("_multi_surgery_flag", "")).lower() in ("true", "1")
    dates = [d for d in str(row.get("_all_bariatric_surgery_dates", "")).split(",") if d.strip()]
    if sd and not ms:
        return "same-day conflicting bariatric CPT"
    if ms and len(dates) >= 2:
        try:
            span = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
            if span <= 2:
                return "surgeries 0-2 days apart (likely billing artifact)"
            elif span <= 180:
                return "surgeries weeks-to-months apart"
            else:
                return "surgeries >180 days apart (likely revision/conversion)"
        except Exception:
            return "multiple surgery dates"
    return "multiple surgery dates"


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)

    required = ["patient_id", "_multi_surgery_flag", "_same_day_ambiguous"]
    missing = set(required) - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    n0 = len(df)
    print(f"Funnel survivors (input): {n0:,}")

    multi = truthy(df["_multi_surgery_flag"])
    sameday = truthy(df["_same_day_ambiguous"])
    drop = multi | sameday

    print(f"\n  flagged multi-surgery:         {multi.sum()}")
    print(f"  flagged same-day ambiguous:    {sameday.sum()}")
    print(f"  flagged BOTH:                  {(multi & sameday).sum()}")
    print(f"  total unique patients to drop: {drop.sum()}")

    excluded = df[drop].copy()
    final = df[~drop].copy()
    if len(excluded):
        excluded["exclusion_reason"] = excluded.apply(categorize, axis=1)

    excluded.to_csv(EXCLUDED_CSV, index=False)
    final.to_csv(FINAL_CSV, index=False)

    print(f"\n{'='*55}\nFINAL COHORT\n{'='*55}")
    print(f"  Funnel survivors:              {n0:,}")
    print(f"  Excluded (multi or same-day):  {drop.sum():,}")
    print(f"  >>> FINAL ANALYTIC COHORT:     {len(final):,} <<<")
    print(f"\n  (Original pipeline reached n=389; a SMALLER number here is expected")
    print(f"   and PI-approved -- the rebuild's multi-surgery definition also catches")
    print(f"   years-apart revision surgeries the original did not.)")

    if len(excluded):
        print(f"\n  Excluded breakdown:")
        print(excluded["exclusion_reason"].value_counts().to_string())
        print(f"\n  Excluded patient detail:")
        cols = ["patient_id", "bariatric_date", "bariatric_cpt_codes_seen",
                "_all_bariatric_surgery_dates", "exclusion_reason"]
        print(excluded[cols].to_string(index=False))


if __name__ == "__main__":
    main()
