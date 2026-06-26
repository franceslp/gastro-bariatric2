#!/usr/bin/env python3
"""
phase1_build_comparator.py  (v2: multi-surgery excluded, surgery_type added)

Builds the PSM comparator pool: bariatric patients who NEVER had gastroparesis,
with the SAME multi-surgery/same-day-ambiguous exclusions applied to the GP cohort,
so the two groups are defined consistently.

Definition (Option B):
  - Bariatric surgery (43775, 43644, 43645, 43846, 43847) in study window
  - NEVER any K31.84 ever
  - NOT in the 376 GP cohort
  - Adult (age >= 18, same approximation as GP cohort: surgery_year - birth_year)
  - EXCLUDE multi-surgery (>1 distinct surgery date) and same-day conflicting CPT
    (mirrors the GP cohort's step-5 exclusions)
  - Diabetes NOT required (PSM covariate)

Output: comparator_pool_raw.csv
Usage:  nohup python3 phase1_build_comparator.py > phase1_log.txt 2>&1 &
"""

import subprocess, re
import pandas as pd
from collections import defaultdict

BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_URI = f"{BASE}/diagnosis.csv"
PROCEDURE_URI = f"{BASE}/procedure.csv"
PATIENT_URI   = f"{BASE}/patient.csv"

GP_COHORT_CSV = "cohort_FINAL_analytic.csv"
OUTPUT_CSV = "comparator_pool_raw.csv"

BARIATRIC = {"43775", "43644", "43645", "43846", "43847"}
SLEEVE = {"43775"}
BYPASS = {"43644", "43645", "43846", "43847"}
STUDY_START = pd.Timestamp("2015-10-01")
STUDY_END   = pd.Timestamp("2025-05-23")


def stream(uri, usecols=None, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", uri], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close(); proc.wait()


def norm_cpt(v):
    s = str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s


def norm_icd(v):
    return re.sub(r"\.", "", str(v).strip().upper())


def parse_date(s):
    d = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if hasattr(d, "isna") and d.isna().all():
        d = pd.to_datetime(s, errors="coerce")
    return d


def surgery_type(codes):
    """Bypass precedence (matches run_PSM logic): any bypass -> bypass; 43775 only -> sleeve."""
    cs = set(codes)
    if cs & BYPASS:
        return "bypass"
    if cs & SLEEVE:
        return "sleeve"
    return None


def main():
    gp = pd.read_csv(GP_COHORT_CSV, dtype={"patient_id": str})
    gp_ids = set(gp["patient_id"])
    print(f"GP cohort to exclude: {len(gp_ids)} patients", flush=True)

    # PASS 1: all K31.84 patients ever
    print("\nPASS 1: diagnosis.csv -> all K31.84 patients", flush=True)
    k_ids = set(); rows = 0; printed = False
    for chunk in stream(DIAGNOSIS_URI, usecols=["patient_id", "code"]):
        rows += len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed = True
        chunk = chunk[chunk["patient_id"].notna()]
        chunk["_i"] = chunk["code"].map(norm_icd)
        k_ids.update(chunk[chunk["_i"] == "K3184"]["patient_id"])
        if rows % 100_000_000 == 0:
            print(f"  ...{rows:,} rows, {len(k_ids):,} K31.84 patients", flush=True)
    print(f"  Patients with any K31.84 ever: {len(k_ids):,}", flush=True)
    exclude = k_ids | gp_ids

    # PASS 2: bariatric patients, collect ALL surgery dates (to detect multi-surgery)
    print("\nPASS 2: procedure.csv -> bariatric surgeries", flush=True)
    surg = defaultdict(list); rows = 0; printed = False
    for chunk in stream(PROCEDURE_URI, usecols=["patient_id", "code", "date"]):
        rows += len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed = True
        chunk = chunk[chunk["patient_id"].notna() & ~chunk["patient_id"].isin(exclude)]
        if chunk.empty:
            if rows % 100_000_000 == 0: print(f"  ...{rows:,} rows", flush=True)
            continue
        chunk = chunk.copy()
        chunk["_c"] = chunk["code"].map(norm_cpt)
        hits = chunk[chunk["_c"].isin(BARIATRIC)]
        if not hits.empty:
            d = parse_date(hits["date"])
            for pid, dt, code in zip(hits["patient_id"], d, hits["_c"]):
                if pd.notna(dt) and STUDY_START <= dt <= STUDY_END:
                    surg[pid].append((dt, code))
        if rows % 100_000_000 == 0:
            print(f"  ...{rows:,} rows, {len(surg):,} candidates", flush=True)
    print(f"  Raw bariatric candidates: {len(surg):,}", flush=True)

    # Apply GP-consistent exclusions: multi-surgery + same-day conflicting
    rows_out = []
    n_multi = n_sameday = n_ambig_type = 0
    for pid, events in surg.items():
        dates = sorted(set(d for d, c in events))
        if len(dates) > 1:
            n_multi += 1
            continue  # multi-surgery: exclude (mirrors GP step 5)
        idx_date = dates[0]
        codes_at_idx = sorted({c for d, c in events if d == idx_date})
        # same-day conflicting: sleeve AND bypass on the index date
        if (set(codes_at_idx) & SLEEVE) and (set(codes_at_idx) & BYPASS):
            n_sameday += 1
            continue  # same-day ambiguous: exclude
        st = surgery_type(codes_at_idx)
        if st is None:
            n_ambig_type += 1
            continue
        rows_out.append({
            "patient_id": pid,
            "bariatric_date": idx_date.date(),
            "bariatric_cpt_codes_seen": ",".join(codes_at_idx),
            "surgery_type": st,
        })
    print(f"  Excluded multi-surgery: {n_multi:,}", flush=True)
    print(f"  Excluded same-day conflicting CPT: {n_sameday:,}", flush=True)
    print(f"  Excluded ambiguous surgery_type: {n_ambig_type:,}", flush=True)
    comp = pd.DataFrame(rows_out)
    print(f"  After exclusions: {len(comp):,}", flush=True)

    # PASS 3: age filter (same approximation as GP)
    print("\nPASS 3: patient.csv -> age >= 18", flush=True)
    comp_ids = set(comp["patient_id"]); yob = {}
    for chunk in stream(PATIENT_URI, usecols=["patient_id", "year_of_birth"]):
        sub = chunk[chunk["patient_id"].isin(comp_ids)]
        for pid, y in zip(sub["patient_id"], sub["year_of_birth"]):
            yob[pid] = y
    comp["year_of_birth"] = comp["patient_id"].map(yob)
    comp["bariatric_date"] = pd.to_datetime(comp["bariatric_date"])
    comp["age_at_surgery_approx"] = comp["bariatric_date"].dt.year - pd.to_numeric(comp["year_of_birth"], errors="coerce")
    before = len(comp)
    comp = comp[(comp["age_at_surgery_approx"] >= 18) & (comp["age_at_surgery_approx"] <= 100)]
    print(f"  Dropped {before - len(comp):,} for age <18 / missing", flush=True)

    comp.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  >>> COMPARATOR POOL: {len(comp):,} patients <<<", flush=True)
    print(f"  surgery_type distribution:\n{comp['surgery_type'].value_counts().to_string()}", flush=True)
    print(f"  Ratio to GP (376): {len(comp)/len(gp_ids):.1f}:1", flush=True)


if __name__ == "__main__":
    main()
