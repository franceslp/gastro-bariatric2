#!/usr/bin/env python3
"""
run_phase2_covariates.py

Sequential wrapper for all Phase 2 covariate collection scripts.
Runs each script in order, logs output to individual log files.
Total runtime: ~4-6 hours.

Run with: nohup python3 run_phase2_covariates.py > phase2_master_log.txt 2>&1 &
"""
import subprocess, sys, time, shutil, os
from datetime import datetime

# ---------------------------------------------------------------------------
# PRE-FLIGHT: create comparator_pool_ready_for_PSM.csv alias
# assemble_and_run_psm.py reads this exact filename. In the rebuild,
# comparator_pool_raw.csv IS the ready-for-PSM file (Phase 1 already applied
# all exclusions: multi-surgery, same-day ambiguous, age, never-K31.84).
# No separate cleaning step needed — just alias the file.
# ---------------------------------------------------------------------------
RAW = "comparator_pool_raw.csv"
ALIAS = "comparator_pool_ready_for_PSM.csv"

if not os.path.exists(RAW):
    print(f"ERROR: {RAW} not found — run phase1_build_comparator.py first.")
    sys.exit(1)

if not os.path.exists(ALIAS):
    shutil.copy(RAW, ALIAS)
    print(f"Created alias: {ALIAS} -> {RAW}")
else:
    print(f"Alias already exists: {ALIAS}")

import pandas as pd
n = len(pd.read_csv(ALIAS, usecols=["patient_id"], dtype=str))
print(f"Comparator pool: {n:,} patients confirmed in {ALIAS}")

SCRIPTS = [
    # Step 1: GP covariates (all in one: diagnosis + labs + meds)
    # ~3 hrs, produces study_covariates_new.csv
    "collect_gp_covariates_new.py",

    # Step 2: Comparator comorbidities (diagnosis.csv)
    # ~25 min, produces psm_covariates_comorbidities.csv
    "collect_psm_covariates_diagnosis.py",

    # Step 3: Comparator meds + dm_circ/dm_other/dyslipidemia (med + diag)
    # ~90 min, produces psm_covariates_comparator_meds_dx.csv
    "collect_comparator_medications_and_dxNEW.py",

    # Step 4: Comparator + GP demographics (patient.csv + diagnosis.csv)
    # ~30 min, produces psm_covariates_comparator_demographics.csv
    "collect_comparator_demographics.py",

    # Step 5: BMI for both groups (vitals_signs.csv)
    # ~45 min, produces gastroparesis_cohort_BMI_at_or_before_surgery.csv
    #          + comparator_pool_ready_for_PSM_with_BMI.csv
    "find_BMI_at_or_before_surgery.py",

    # Step 6: Diabetes duration for both groups (diagnosis.csv)
    # ~25 min, produces psm_covariates_true_diabetes_duration.csv
    "collect_true_diabetes_duration.py",

    # Step 7: Comparator labs/A1c (lab_result.csv)
    # ~30 min, produces psm_covariates_labs.csv
    "collect_psm_covariates_labs.py",
]

def run(script):
    log = script.replace(".py", "_log.txt")
    print(f"\n{'='*60}", flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING: {script}", flush=True)
    print(f"  Log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        result = subprocess.run(
            [sys.executable, script],
            stdout=f, stderr=subprocess.STDOUT
        )
    elapsed = (time.time() - t0) / 60
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {script} ({elapsed:.1f} min)", flush=True)
    if result.returncode != 0:
        print(f"  Last 20 lines of {log}:", flush=True)
        with open(log) as f:
            lines = f.readlines()
        for line in lines[-20:]:
            print(f"    {line.rstrip()}", flush=True)
        print("  STOPPING — fix error before continuing.", flush=True)
        sys.exit(1)
    return elapsed

total = 0
for s in SCRIPTS:
    total += run(s)
print(f"\n{'='*60}")
print(f"Phase 2 complete. Total runtime: {total:.1f} min")
print("Next: run assemble_and_run_psm.py (Phase 3)")
