#!/usr/bin/env python3
"""Resume Phase 2 from step 2 (after collect_gp_covariates_new.py succeeded)."""
import subprocess, sys, time, os
from datetime import datetime
import pandas as pd

SCRIPTS = [
    "collect_psm_covariates_diagnosis.py",
    "collect_comparator_medications_and_dxNEW.py",
    "collect_comparator_demographics.py",
    "find_BMI_at_or_before_surgery.py",
    "collect_true_diabetes_duration.py",
    "collect_psm_covariates_labs.py",
]

def run(script):
    log = script.replace(".py", "_log.txt")
    print(f"\n{'='*60}", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: {script}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        result = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {script} ({elapsed:.1f} min)", flush=True)
    if result.returncode != 0:
        with open(log) as f: lines = f.readlines()
        for line in lines[-20:]: print(f"    {line.rstrip()}", flush=True)
        print("  STOPPING.", flush=True)
        sys.exit(1)
    return elapsed

if not os.path.exists("study_covariates_new.csv"):
    print("ERROR: study_covariates_new.csv not found.")
    sys.exit(1)
n = len(pd.read_csv("study_covariates_new.csv", dtype=str))
print(f"GP covariates confirmed: {n} patients")
if n < 370:
    print(f"ERROR: expected ~376, got {n}.")
    sys.exit(1)

if not os.path.exists("comparator_pool_ready_for_PSM.csv"):
    print("ERROR: comparator_pool_ready_for_PSM.csv not found.")
    sys.exit(1)
nc = len(pd.read_csv("comparator_pool_ready_for_PSM.csv", dtype=str))
print(f"Comparator pool confirmed: {nc} patients")
if nc < 7000:
    print(f"WARNING: comparator pool is {nc}, expected ~7027.")
    sys.exit(1)

total = sum(run(s) for s in SCRIPTS)
print(f"\nPhase 2 steps 2-7 complete. Total: {total:.1f} min")
print("Next: verify BMI source fix in assemble_and_run_psm.py, then run it.")
