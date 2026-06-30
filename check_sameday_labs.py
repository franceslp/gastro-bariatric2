#!/usr/bin/env python3
"""
check_sameday_labs.py

For the 376 GP patients, checks how many had an A1c or BMI lab value dated
EXACTLY on the surgery date (day 0) — these would have been excluded from
baseline_a1c / preoperative_bmi under the 1-365 day pre-surgery window
(diff >= 1 excludes day 0).

This is a diagnostic check, not a pipeline change. Answers: "how many
patients were affected by the same-day exclusion rule?"

Scans lab_result.csv once for both LOINC code sets.
"""
import subprocess
import pandas as pd

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE = f"{GCS_BASE}/lab_result.csv"
CHUNK = 500_000

A1C_LOINC = {"4548-4", "17856-6", "4549-2"}
BMI_LOINC = {"39156-5"}
A1C_MIN, A1C_MAX = 2, 20
BMI_MIN, BMI_MAX = 10, 100

def parse_dates(series):
    s = series.fillna("").astype(str)
    r = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    m = r.isna() & (s.str.strip() != "")
    if m.any():
        r.loc[m] = pd.to_datetime(s.loc[m], format="mixed", errors="coerce")
    return r

def stream(path, usecols):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(proc.stdout, usecols=usecols,
                                  dtype=str, chunksize=CHUNK):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

print("Loading GP cohort (376 patients) and surgery dates...")
gp = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
gp["surg_dt"] = pd.to_datetime(gp["bariatric_date"], errors="coerce")
all_ids = set(gp["patient_id"])
surgery_dt = dict(zip(gp["patient_id"], gp["surg_dt"]))
print(f"  {len(all_ids)} patients loaded")

# Also load current baseline_a1c/preoperative_bmi to know who is ALREADY missing
# (so we can distinguish "missing because no lab at all" vs
#  "missing because only a same-day lab existed")
try:
    cov = pd.read_csv("study_covariates_new.csv", dtype={"patient_id": str})
    current_a1c_missing = set(cov[cov["baseline_a1c"].isna()]["patient_id"])
except FileNotFoundError:
    current_a1c_missing = set()
    print("  WARNING: study_covariates_new.csv not found, skipping cross-check")

try:
    bmi_file = pd.read_csv("gastroparesis_cohort_BMI_at_or_before_surgery.csv",
                           dtype={"patient_id": str})
    current_bmi_missing = set(
        bmi_file[bmi_file["BMI_at_or_before_surgery"].isna()]["patient_id"])
except FileNotFoundError:
    current_bmi_missing = set()
    print("  WARNING: BMI file not found, skipping cross-check")

# Store same-day lab hits
sameday_a1c = {}  # pid -> value
sameday_bmi = {}  # pid -> value

print("\nScanning lab_result.csv for same-day (day 0) A1c and BMI...")
rows = 0
for chunk in stream(LAB_FILE,
                    ["patient_id", "code_system", "code", "date", "lab_result_num_val"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue
    chunk["code_system"] = chunk["code_system"].str.upper().str.strip()
    chunk = chunk[chunk["code_system"] == "LOINC"]
    chunk["code"] = chunk["code"].str.strip()
    relevant_codes = A1C_LOINC | BMI_LOINC
    chunk = chunk[chunk["code"].isin(relevant_codes)]
    if chunk.empty:
        continue
    chunk["dt"] = parse_dates(chunk["date"])
    chunk = chunk[chunk["dt"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days"] = (chunk["dt"] - chunk["surg"]).dt.days
    # ONLY day 0 (exactly the surgery date)
    chunk = chunk[chunk["days"] == 0]
    if chunk.empty:
        continue
    chunk["val"] = pd.to_numeric(chunk["lab_result_num_val"], errors="coerce")
    for pid, code, val in zip(chunk["patient_id"], chunk["code"], chunk["val"]):
        if code in A1C_LOINC and A1C_MIN <= val <= A1C_MAX:
            sameday_a1c[pid] = val
        elif code in BMI_LOINC and BMI_MIN <= val <= BMI_MAX:
            sameday_bmi[pid] = val
    if rows % 500_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

print("="*70)
print("SAME-DAY (SURGERY DATE) LAB RESULTS — Diagnostic Check")
print("="*70)
print(f"\nA1c on surgery date: {len(sameday_a1c)} patients")
print(f"BMI on surgery date: {len(sameday_bmi)} patients")

# Cross-check: of these, how many are CURRENTLY missing baseline_a1c/bmi?
# (i.e., the same-day value was their ONLY nearby lab, so excluding it
#  pushed them into "missing")
if current_a1c_missing:
    affected_a1c = set(sameday_a1c.keys()) & current_a1c_missing
    print(f"\nOf the {len(sameday_a1c)} with same-day A1c, "
          f"{len(affected_a1c)} are CURRENTLY MISSING baseline_a1c")
    print("  (meaning: their ONLY nearby A1c was on surgery day itself,")
    print("   and excluding it left them with no valid baseline value)")
    if affected_a1c:
        print(f"  Affected patient IDs: {sorted(affected_a1c)}")

if current_bmi_missing:
    affected_bmi = set(sameday_bmi.keys()) & current_bmi_missing
    print(f"\nOf the {len(sameday_bmi)} with same-day BMI, "
          f"{len(affected_bmi)} are CURRENTLY MISSING preoperative_bmi")
    print("  (meaning: their ONLY nearby BMI was on surgery day itself,")
    print("   and excluding it left them with no valid baseline value)")
    if affected_bmi:
        print(f"  Affected patient IDs: {sorted(affected_bmi)}")

# Save for reference
pd.DataFrame([{"patient_id":p,"sameday_a1c":v} for p,v in sameday_a1c.items()]).to_csv(
    "sameday_a1c_check.csv", index=False)
pd.DataFrame([{"patient_id":p,"sameday_bmi":v} for p,v in sameday_bmi.items()]).to_csv(
    "sameday_bmi_check.csv", index=False)
print("\nWrote: sameday_a1c_check.csv, sameday_bmi_check.csv")
