#!/usr/bin/env python3
"""
step2_exclusion_funnel.py  (self-contained: own diagnosis + procedure scans)

Stepwise exclusion funnel on master_cohort_rebuilt_FINAL.csv.
Writes one CSV of SURVIVORS after each step. Scans diagnosis.csv (full E10/E11
date lists) and procedure.csv (full GES date lists) itself, so no step relies on
a pre-derived single-date column.

Steps (each operates on survivors of the previous):
  0  start: all rebuilt cohort patients
  1  age >= 18 at surgery                          -> funnel_step1_age.csv
  2  >=1 K31.84 in [surgery-365, surgery)          -> funnel_step2_k3184_1yr.csv
  3  >=1 E10/E11 in [surgery-365, surgery)         -> funnel_step3_e10e11_1yr.csv
  4  >=1 GES that is < surgery AND < some K31.84    -> funnel_step4_ges.csv

CONVENTIONS (documented):
  - All windows use STRICT < surgery (same-day-as-surgery events do NOT count),
    consistent with 'closest_K31_84_strictly_before_surgery'.
  - K31.84 dates come from the full all_K31_84_dates list in the master file.
  - E10/E11 and GES dates are scanned fresh from raw files (full lists, not
    derived closest-date columns).
  - Multi-surgery + same-day-ambiguous exclusions are applied LATER, not here.

Usage:  nohup python3 step2_exclusion_funnel.py > step2_funnel_log.txt 2>&1 &
"""

import subprocess
import pandas as pd
from collections import defaultdict

BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_URI = f"{BASE}/diagnosis.csv"
PROCEDURE_URI = f"{BASE}/procedure.csv"
MASTER_CSV = "master_cohort_rebuilt_FINAL.csv"
GES_CODES = {"78264", "78265", "78266"}
WINDOW_DAYS = 365


def stream(uri, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", uri], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close(); proc.wait()


def parse_date(s):
    d = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if hasattr(d, "isna") and d.isna().all():
        d = pd.to_datetime(s, errors="coerce")
    return d


def norm_cpt(v):
    s = str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s


def norm_icd(v):
    import re
    return re.sub(r"\.", "", str(v).strip().upper())


def date_list(s):
    ds = pd.to_datetime([x.strip() for x in str(s).split(",") if x.strip()], errors="coerce")
    return [d for d in ds if pd.notna(d)]


def main():
    df = pd.read_csv(MASTER_CSV, dtype=str)
    df["_surg"] = pd.to_datetime(df["bariatric_date"], errors="coerce")
    n0 = len(df)
    ids = set(df["patient_id"])
    surg_map = dict(zip(df["patient_id"], df["_surg"]))
    print(f"Step 0 — start: {n0:,} patients", flush=True)

    # ---- SCAN 1: diagnosis.csv -> full E10/E11 date list per patient ----
    print("\nScanning diagnosis.csv for ALL E10/E11 dates...", flush=True)
    e_dates = defaultdict(list); rows = 0; printed = False
    for chunk in stream(DIAGNOSIS_URI):
        rows += len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed = True
        sub = chunk[chunk["patient_id"].isin(ids)].copy()
        if sub.empty:
            if rows % 50_000_000 == 0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub["_i"] = sub["code"].map(norm_icd)
        e = sub[sub["_i"].str.startswith(("E10", "E11"), na=False)]
        if not e.empty:
            d = parse_date(e["date"])
            for pid, dt in zip(e["patient_id"], d):
                if pd.notna(dt): e_dates[pid].append(dt)
        if rows % 50_000_000 == 0:
            print(f"  ...{rows:,} rows, {len(e_dates):,} E-code pts", flush=True)
    print(f"  E10/E11 dates collected for {len(e_dates):,} patients", flush=True)

    # ---- SCAN 2: procedure.csv -> full GES date list per patient ----
    print("\nScanning procedure.csv for ALL GES dates...", flush=True)
    ges_dates = defaultdict(list); rows = 0; printed = False
    for chunk in stream(PROCEDURE_URI):
        rows += len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed = True
        sub = chunk[chunk["patient_id"].isin(ids)].copy()
        if sub.empty:
            if rows % 50_000_000 == 0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub["_c"] = sub["code"].map(norm_cpt)
        g = sub[sub["_c"].isin(GES_CODES)]
        if not g.empty:
            d = parse_date(g["date"])
            for pid, dt in zip(g["patient_id"], d):
                if pd.notna(dt): ges_dates[pid].append(dt)
        if rows % 50_000_000 == 0:
            print(f"  ...{rows:,} rows, {len(ges_dates):,} GES pts", flush=True)
    print(f"  GES dates collected for {len(ges_dates):,} patients", flush=True)

    # ---- STEP 1: age >= 18 ----
    def age_ok(r):
        try:
            return (r["_surg"].year - int(r["year_of_birth"])) >= 18
        except Exception:
            return False
    s1 = df[df.apply(age_ok, axis=1)].copy()
    print(f"\nStep 1 — age>=18: {len(s1):,}  (lost {n0-len(s1):,})", flush=True)
    s1.drop(columns=["_surg"]).to_csv("funnel_step1_age.csv", index=False)

    # ---- STEP 2: K31.84 within 365d before surgery (full list from master) ----
    def k_within(r):
        surg = r["_surg"]
        if pd.isna(surg): return False
        lo = surg - pd.Timedelta(days=WINDOW_DAYS)
        return any(lo <= d < surg for d in date_list(r.get("all_K31_84_dates", "")))
    s2 = s1[s1.apply(k_within, axis=1)].copy()
    print(f"Step 2 — K31.84 within {WINDOW_DAYS}d before surgery: {len(s2):,}  (lost {len(s1)-len(s2):,})", flush=True)
    s2.drop(columns=["_surg"]).to_csv("funnel_step2_k3184_1yr.csv", index=False)

    # ---- STEP 3: E10/E11 within 365d before surgery (full scanned list) ----
    def e_within(r):
        surg = r["_surg"]
        if pd.isna(surg): return False
        lo = surg - pd.Timedelta(days=WINDOW_DAYS)
        return any(lo <= d < surg for d in e_dates.get(r["patient_id"], []))
    s3 = s2[s2.apply(e_within, axis=1)].copy()
    print(f"Step 3 — E10/E11 within {WINDOW_DAYS}d before surgery: {len(s3):,}  (lost {len(s2)-len(s3):,})", flush=True)
    s3.drop(columns=["_surg"]).to_csv("funnel_step3_e10e11_1yr.csv", index=False)

    # ---- STEP 4: GES < surgery AND < at least one K31.84 (full lists) ----
    def ges_ok(r):
        surg = r["_surg"]
        if pd.isna(surg): return False
        gds = ges_dates.get(r["patient_id"], [])
        kds = date_list(r.get("all_K31_84_dates", ""))
        if not gds or not kds: return False
        for g in gds:
            if g < surg and any(g < k for k in kds):
                return True
        return False
    s4 = s3[s3.apply(ges_ok, axis=1)].copy()
    print(f"Step 4 — GES before surgery & before >=1 K31.84: {len(s4):,}  (lost {len(s3)-len(s4):,})", flush=True)
    s4.drop(columns=["_surg"]).to_csv("funnel_step4_ges.csv", index=False)

    # ---- CONSORT summary table (one row per step) ----
    summary = pd.DataFrame([
        {"step": "0_start",            "criterion": "rebuilt cohort",                       "n_remaining": n0,      "n_lost": 0},
        {"step": "1_age",              "criterion": "age >= 18 at surgery",                 "n_remaining": len(s1), "n_lost": n0-len(s1)},
        {"step": "2_k3184_1yr",        "criterion": "K31.84 within 365d before surgery",    "n_remaining": len(s2), "n_lost": len(s1)-len(s2)},
        {"step": "3_e10e11_1yr",       "criterion": "E10/E11 within 365d before surgery",   "n_remaining": len(s3), "n_lost": len(s2)-len(s3)},
        {"step": "4_ges",             "criterion": "GES before surgery & before >=1 K31.84","n_remaining": len(s4), "n_lost": len(s3)-len(s4)},
    ])
    summary.to_csv("funnel_consort_summary.csv", index=False)

    print("\n" + "="*60 + "\nFUNNEL SUMMARY\n" + "="*60, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\n  (Original funnel reference: 1118 -> 1117 -> 907 -> 879 -> 396)", flush=True)
    print(f"  Multi-surgery + same-day-ambiguous applied AFTER this step.", flush=True)


if __name__ == "__main__":
    main()
