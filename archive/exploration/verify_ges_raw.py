#!/usr/bin/env python3
"""
verify_ges_raw.py

Independent verification of GES capture. Scans raw procedure.csv from GCS and
counts how many of the 1,118 base-cohort patients have ANY GES CPT code
(78264 / 78265 / 78266), normalizing float-formatted codes and whitespace.

This is independent of the has_GES flag, so it confirms whether the upstream
pipeline missed any GES patients. It counts ANY historical GES (no timing
filter) and does NOT pre-filter on code_system -- the code_system breakdown is
reported so a mismatch (e.g. CPT4 vs CPT) would be visible rather than hidden.

Expected result: 511 unique patients.

Usage (run where funnel_1_all_patients_1118.csv lives):
    nohup python3 verify_ges_raw.py > ges_verify_log.txt 2>&1 &
"""

import subprocess
import pandas as pd

COHORT_FILE = "funnel_1_all_patients_1118.csv"
PROCEDURE_URI = (
    "gs://test-skynet-lh/joseph-sujka/"
    "trinetx-gastroparesis-dyspepsia/procedure.csv"
)
GES_CODES = {"78264", "78265", "78266"}
EXPECTED = 511


def normalize_code(value):
    """Strip whitespace and a trailing '.0' so float-formatted codes match."""
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def main():
    cohort = pd.read_csv(COHORT_FILE, dtype={"patient_id": str})
    cohort_ids = set(cohort["patient_id"])
    print(f"Cohort size: {len(cohort_ids):,} patients", flush=True)

    proc = subprocess.Popen(
        ["gsutil", "cat", PROCEDURE_URI],
        stdout=subprocess.PIPE,
    )

    found = []
    rows = 0
    code_col = None
    for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=500_000):
        rows += len(chunk)

        if code_col is None:
            possible = ["code", "cpt_code", "procedure_code", "code_value"]
            code_col = next(
                (c for c in chunk.columns if c.lower() in possible), None
            )
            if code_col is None:
                raise ValueError(
                    f"No recognized code column. Columns: {list(chunk.columns)}"
                )

        sub = chunk[chunk["patient_id"].isin(cohort_ids)]
        if not sub.empty:
            sub = sub.copy()
            sub["_norm"] = sub[code_col].map(normalize_code)
            hit = sub[sub["_norm"].isin(GES_CODES)]
            if not hit.empty:
                found.append(hit)

        if rows % 50_000_000 == 0:
            print(f"  ...{rows:,} rows scanned", flush=True)

    proc.stdout.close()
    proc.wait()

    print(f"\nTotal rows scanned: {rows:,}")
    if found:
        res = pd.concat(found)
        n = res["patient_id"].nunique()
        print(
            f"Unique cohort patients with ANY GES code, any date "
            f"(raw capture check): {n}"
        )
        print("\nCode breakdown:")
        print(res["_norm"].value_counts())
        cs = [c for c in res.columns if "system" in c.lower()]
        if cs:
            print("\ncode_system breakdown (NOT filtered -- inspect for CPT4/blank):")
            print(res[cs[0]].value_counts(dropna=False))
        print(f"\nMatch expected ({EXPECTED}): {n == EXPECTED}")
        if n != EXPECTED:
            print(
                f"Difference: {n - EXPECTED:+d}  -- if positive, upstream "
                f"missed patients; check code_system breakdown above"
            )
    else:
        print("No GES records found")


if __name__ == "__main__":
    main()
