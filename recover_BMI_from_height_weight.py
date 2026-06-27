#!/usr/bin/env python3
"""
recover_BMI_from_height_weight.py

Recovers additional BMI coverage by deriving BMI from recorded height and
weight for patients who lack a directly-recorded BMI (LOINC 39156-5). This
increases BMI capture using a standard EHR-derived BMI approach (computing
BMI from measured height and weight), keeping the same 1-365 day pre-surgery
window used for all other covariates.

VITALS LOINC CODES (confirmed present in vitals_signs.csv):
  39156-5  BMI            (kg/m2)    -- direct, preferred when available
  8302-2   body height    ([in_us], inches)
  29463-7  body weight    ([lb_av], pounds)
  3141-9   body weight    ([lb_av], pounds)  -- second weight code

BMI formula (imperial): BMI = 703 * weight_lb / (height_in)^2

PAIRING (Issue 1 fix): weight is the time-sensitive measure, so we take the
MOST RECENT weight in the window, then pair it with the height measurement
CLOSEST IN TIME to that weight (adult height is ~stable, so a slightly older
height is acceptable; weight must be recent). Height and weight need not come
from the same encounter, but are matched by proximity rather than taken
independently.

PLAUSIBILITY (Issue 3 fix): height 36-96 in, weight 50-700 lb, BMI 10-100.

WINDOW: 1-365 days before surgery (diff >= 1), consistent with the A1c and
original BMI collection. Same-day (diff=0) excluded for consistency across
covariates. (Whether to include same-day is a methods question for the PI.)

OUTPUT for each cohort:
  patient_id, bmi_direct, bmi_computed, BMI_at_or_before_surgery, bmi_source
    bmi_source in {"direct", "computed", "missing"}
    BMI_at_or_before_surgery = direct if present else computed

Also prints a VALIDATION report: for patients with BOTH direct BMI and
height+weight, compares computed vs recorded to confirm the formula/units.

OUTPUT FILENAMES (Issue 4): writes to *_recovered.csv (does NOT overwrite the
existing validated BMI files). Point the PSM script at these after confirming
the validation report looks good.
"""
import subprocess
import numpy as np
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
VITALS_FILE = f"{GCS_BASE}/vitals_signs.csv"
CHUNK = 500_000
WINDOW_DAYS = 365

BMI_LOINC    = {"39156-5"}
HEIGHT_LOINC = {"8302-2"}
WEIGHT_LOINC = {"29463-7", "3141-9"}
ALL_CODES = BMI_LOINC | HEIGHT_LOINC | WEIGHT_LOINC

# Plausibility bounds (Issue 3)
H_MIN, H_MAX = 36, 96     # inches (3-8 ft)
W_MIN, W_MAX = 50, 700    # pounds
BMI_MIN, BMI_MAX = 10, 100

COHORTS = [
    ("GP",         "cohort_FINAL_analytic.csv",         "bariatric_date", "gp_BMI_recovered.csv"),
    ("comparator", "comparator_pool_ready_for_PSM.csv", "bariatric_date", "comparator_BMI_recovered.csv"),
]

def parse_dates(series):
    series = series.fillna("").astype(str)
    r = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    m = r.isna() & (series.str.strip() != "")
    if m.any():
        r.loc[m] = pd.to_datetime(series.loc[m], format="mixed", errors="coerce")
    return r

def stream(path, usecols):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(proc.stdout, usecols=usecols, dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

cohort_info = {}
all_ids = set()
surgery_global = {}
for label, csv, datecol, _ in COHORTS:
    df = pd.read_csv(csv, dtype={"patient_id": str})
    df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
    lookup = dict(zip(df["patient_id"], df[datecol]))
    cohort_info[label] = {"ids": set(df["patient_id"].dropna()), "surgery": lookup}
    all_ids |= cohort_info[label]["ids"]
    surgery_global.update(lookup)
    print(f"{label}: {len(cohort_info[label]['ids']):,} patients")

# Issue 1 fix: keep ALL height and weight measurements (as lists of (val,date)),
# plus the single most-recent direct BMI. We pair height/weight after scanning.
store = {pid: {"bmi": None, "height": [], "weight": []} for pid in all_ids}

print("\nScanning vitals_signs.csv...")
rows = 0
for chunk in stream(VITALS_FILE,
                    ["patient_id", "code_system", "code", "date", "value", "units_of_measure"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)]
    if chunk.empty:
        continue
    chunk["code"] = chunk["code"].str.strip()
    chunk = chunk[chunk["code"].isin(ALL_CODES)]
    if chunk.empty:
        continue
    chunk["date"] = parse_dates(chunk["date"])
    chunk["val"]  = pd.to_numeric(chunk["value"], errors="coerce")
    chunk = chunk[chunk["val"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_global)
    chunk = chunk[chunk["surg"].notna()]
    chunk["diff"] = (chunk["surg"] - chunk["date"]).dt.days
    chunk = chunk[(chunk["diff"] >= 1) & (chunk["diff"] <= WINDOW_DAYS)]
    if chunk.empty:
        continue
    for pid, code, val, date in zip(chunk["patient_id"], chunk["code"], chunk["val"], chunk["date"]):
        if code in BMI_LOINC:
            cur = store[pid]["bmi"]
            if cur is None or date > cur[1]:
                store[pid]["bmi"] = (val, date)
        elif code in HEIGHT_LOINC:
            if H_MIN <= val <= H_MAX:           # Issue 3
                store[pid]["height"].append((val, date))
        else:  # weight
            if W_MIN <= val <= W_MAX:           # Issue 3
                store[pid]["weight"].append((val, date))
    if rows % 50_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

def pair_height_weight(hlist, wlist):
    """Most recent weight; height closest in time to that weight's date."""
    if not wlist:
        return None
    w_val, w_date = max(wlist, key=lambda t: t[1])   # most recent weight
    if not hlist:
        return None
    h_val, h_date = min(hlist, key=lambda t: abs((t[1] - w_date).days))  # closest height
    return h_val, w_val

for label, csv, datecol, outcsv in COHORTS:
    ids = cohort_info[label]["ids"]
    recs = []
    val_direct, val_computed = [], []
    for pid in ids:
        s = store[pid]
        bmi_direct = s["bmi"][0] if s["bmi"] else np.nan
        bmi_computed = np.nan
        pair = pair_height_weight(s["height"], s["weight"])
        if pair:
            h_in, w_lb = pair
            b = 703.0 * w_lb / (h_in ** 2)
            if BMI_MIN <= b <= BMI_MAX:
                bmi_computed = round(b, 2)
        if not np.isnan(bmi_direct) and not np.isnan(bmi_computed):
            val_direct.append(bmi_direct); val_computed.append(bmi_computed)
        if not np.isnan(bmi_direct):
            final, src = bmi_direct, "direct"
        elif not np.isnan(bmi_computed):
            final, src = bmi_computed, "computed"
        else:
            final, src = np.nan, "missing"
        recs.append({"patient_id": pid, "bmi_direct": bmi_direct,
                     "bmi_computed": bmi_computed,
                     "BMI_at_or_before_surgery": final, "bmi_source": src})
    out = pd.DataFrame(recs)

    n = len(out)
    n_d = (out["bmi_source"] == "direct").sum()
    n_c = (out["bmi_source"] == "computed").sum()
    n_m = (out["bmi_source"] == "missing").sum()
    print(f"=== {label} ===")
    print(f"  direct BMI:    {n_d} ({100*n_d/n:.1f}%)")
    print(f"  + computed:    {n_c}  -> total coverage {100*(n_d+n_c)/n:.1f}%")
    print(f"  still missing: {n_m} ({100*n_m/n:.1f}%)")
    if val_direct:
        vd = np.array(val_direct); vc = np.array(val_computed); diff = vc - vd
        print(f"  VALIDATION (n={len(vd)} with both):")
        print(f"    mean diff (computed-direct): {diff.mean():+.2f} (SD {diff.std():.2f})")
        print(f"    within +/-1.0: {np.mean(np.abs(diff)<=1)*100:.1f}% | "
              f"+/-2.0: {np.mean(np.abs(diff)<=2)*100:.1f}%")
        print(f"    correlation: {np.corrcoef(vd, vc)[0,1]:.4f}")
    else:
        print("  VALIDATION: no patients with both (cannot validate)")
    out.to_csv(outcsv, index=False)
    print(f"  wrote {outcsv}\n")

print("Done. Review the VALIDATION report before using computed values.")
