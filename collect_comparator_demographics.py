"""
collect_comparator_demographics.py

Collects demographics and diabetes type flags for the comparator pool.
These exist for the GP cohort in funnel_6_final_cohort.csv but are absent
from comparator_pool_ready_for_PSM.csv.

Variables collected:
  From patient.csv:
    race      → race_white (1/0), race_black (1/0)
    ethnicity → ethnicity_hispanic (1/0)

  From diagnosis.csv (ICD-10-CM, before surgery):
    t1dm  → any E10.x before surgery
    t2dm  → any E11.x before surgery

ENCODING NOTES:
  race_white       = race contains "white" OR "caucasian"
  race_black       = race contains "black" OR "african"
  ethnicity_hispanic = ethnicity contains "hispanic" OR "latino" OR "spanish"
  Missing race/ethnicity → filled as 0 (unknown treated as not-flagged),
    consistent with GP cohort encoding. Change if GP cohort uses a different
    missing strategy.

  T1DM + T2DM overlap is possible (miscoding). QA prints overlap count.
  Keep as separate flags for PSM rather than forcing mutual exclusivity.

Output: psm_covariates_comparator_demographics.csv
"""

import subprocess
import time
import pandas as pd

GCS_BASE       = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PATIENT_FILE   = f"{GCS_BASE}/patient.csv"
DIAGNOSIS_FILE = f"{GCS_BASE}/diagnosis.csv"

COMPARATOR_CSV  = "comparator_pool_raw.csv"
GP_COHORT_CSV   = "cohort_FINAL_analytic.csv"
OUTPUT_CSV      = "psm_covariates_comparator_demographics.csv"

print(">>> SCRIPT VERSION: collect_comparator_demographics_v2 <<<")

# ---------------------------------------------------------------------------
# Load comparator pool
# ---------------------------------------------------------------------------
comp = pd.read_csv(COMPARATOR_CSV, dtype={"patient_id": str},
                   usecols=["patient_id", "bariatric_date"])
comp = comp.drop_duplicates(subset="patient_id")
comp["surgery_dt"] = pd.to_datetime(comp["bariatric_date"], errors="coerce")

surgery_lookup = dict(zip(comp["patient_id"], comp["surgery_dt"]))
comp_ids = set(comp["patient_id"].dropna())
print(f"Comparator patients: {len(comp_ids):,}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def stream_gcs_csv(path, usecols, chunksize=500_000):
    proc = subprocess.Popen(["gsutil", "cat", path], stdout=subprocess.PIPE)
    try:
        for chunk in pd.read_csv(
            proc.stdout, usecols=usecols, dtype=str, chunksize=chunksize
        ):
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()

# ===========================================================================
# PASS 1 — patient.csv
# ===========================================================================
print("\n--- PASS 1: patient.csv (race, ethnicity) ---")
t0 = time.time()

demo_rows = []
rows_seen = 0
for chunk in stream_gcs_csv(PATIENT_FILE, usecols=["patient_id", "race", "ethnicity"]):
    rows_seen += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(comp_ids)].copy()
    if not chunk.empty:
        demo_rows.append(chunk)

print(f"Done: {rows_seen:,} rows scanned in {(time.time()-t0)/60:.1f} min")

demo_df = pd.concat(demo_rows, ignore_index=True) if demo_rows else pd.DataFrame()
print(f"Matched {len(demo_df):,} comparator rows")

if not demo_df.empty:
    # FIX 5: Check for multiple rows per patient before dedup
    dupes = demo_df["patient_id"].duplicated().sum()
    if dupes:
        print(f"  Note: {dupes:,} duplicate patient rows in patient.csv — keeping first")
    demo_df = demo_df.drop_duplicates(subset="patient_id", keep="first")

    # Print raw values — keep this, it catches TriNetX string quirks
    print("\nRaw race values (top 15):")
    print(demo_df["race"].value_counts().head(15).to_string())
    print("\nRaw ethnicity values (top 10):")
    print(demo_df["ethnicity"].value_counts().head(10).to_string())

    # FIX 1: race_white includes "caucasian"
    # FIX 2: ethnicity_hispanic includes "spanish"
    race_lower = demo_df["race"].str.strip().str.lower().fillna("")
    demo_df["race_white"] = race_lower.str.contains(
        "white|caucasian", na=False
    ).astype(int)
    demo_df["race_black"] = race_lower.str.contains(
        "black|african", na=False
    ).astype(int)

    eth_lower = demo_df["ethnicity"].str.strip().str.lower().fillna("")
    demo_df["ethnicity_hispanic"] = eth_lower.str.contains(
        "hispanic|latino|spanish", na=False
    ).astype(int)

    # Print GP cohort raw race values for direct comparison
    try:
        gp = pd.read_csv(GP_COHORT_CSV, usecols=["race", "ethnicity"])
        print("\nGP cohort raw race values (top 10) — for encoding comparison:")
        print(gp["race"].value_counts().head(10).to_string())
        print("\nGP cohort raw ethnicity values (top 10):")
        print(gp["ethnicity"].value_counts().head(10).to_string())
    except Exception:
        print("  (GP cohort file not available for comparison)")

else:
    print("WARNING: No demographic rows found — filling race/ethnicity as 0")
    demo_df = pd.DataFrame({"patient_id": list(comp_ids)})
    for col in ["race_white", "race_black", "ethnicity_hispanic"]:
        demo_df[col] = 0

# ===========================================================================
# PASS 2 — diagnosis.csv (T1DM / T2DM flags before surgery)
# ===========================================================================
print("\n--- PASS 2: diagnosis.csv (T1DM / T2DM flags) ---")
t0 = time.time()
rows_seen = chunk_num = 0

t1dm_patients = set()
t2dm_patients = set()

for chunk in stream_gcs_csv(
    DIAGNOSIS_FILE,
    usecols=["patient_id", "code_system", "code", "date"],
):
    chunk_num += 1
    rows_seen += len(chunk)

    chunk = chunk[chunk["patient_id"].isin(comp_ids)].copy()
    if chunk.empty:
        continue

    chunk["code_system"] = chunk["code_system"].str.strip()
    icd = chunk[chunk["code_system"] == "ICD-10-CM"].copy()
    if icd.empty:
        continue

    icd["code"] = icd["code"].str.strip()
    icd = icd[icd["code"].str.startswith(("E10", "E11"))]
    if icd.empty:
        continue

    icd["date"] = pd.to_datetime(icd["date"], errors="coerce")
    icd = icd[icd["date"].notna()]
    icd["surgery_dt"] = icd["patient_id"].map(surgery_lookup)
    icd = icd[icd["date"] < icd["surgery_dt"]]
    if icd.empty:
        continue

    t1dm_patients.update(
        icd.loc[icd["code"].str.startswith("E10"), "patient_id"].unique()
    )
    t2dm_patients.update(
        icd.loc[icd["code"].str.startswith("E11"), "patient_id"].unique()
    )

    if chunk_num % 50 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  ...{rows_seen:,} rows | {elapsed:.1f} min | "
              f"T1DM: {len(t1dm_patients):,} | T2DM: {len(t2dm_patients):,}")

print(f"Done: {rows_seen:,} rows in {(time.time()-t0)/60:.1f} min")
print(f"T1DM patients: {len(t1dm_patients):,}")
print(f"T2DM patients: {len(t2dm_patients):,}")

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------
out_df = pd.DataFrame({"patient_id": list(comp_ids)})

out_df = out_df.merge(
    demo_df[["patient_id", "race_white", "race_black", "ethnicity_hispanic"]],
    on="patient_id", how="left"
)

out_df["t1dm"] = out_df["patient_id"].isin(t1dm_patients).astype(int)
out_df["t2dm"] = out_df["patient_id"].isin(t2dm_patients).astype(int)

# FIX 3: Fill missing race/ethnicity as 0 (unknown = not flagged)
# Consistent with GP cohort where missing demographics are 0, not NaN.
# If GP cohort uses a different strategy, change this.
race_eth_cols = ["race_white", "race_black", "ethnicity_hispanic"]
out_df[race_eth_cols] = out_df[race_eth_cols].fillna(0).astype(int)

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------
print("\nQA — completeness and prevalence:")
# FIX (added): full missingness check
print("\nMissingness (should be 0 after fillna):")
print(out_df.isna().mean().to_string())

demo_cols = ["race_white", "race_black", "ethnicity_hispanic", "t1dm", "t2dm"]
print("\nPrevalence:")
for col in demo_cols:
    n   = out_df[col].sum()
    pct = n / len(out_df) * 100
    print(f"  {col:<22}: {n:,} ({pct:.1f}%)")

# FIX 4: T1+T2 overlap check
both = ((out_df["t1dm"] == 1) & (out_df["t2dm"] == 1)).sum()
print(f"\nT1DM + T2DM overlap (both=1): {both:,} patients "
      f"({both/len(out_df)*100:.1f}%) — expected due to TriNetX miscoding")

# GP cross-check for race/eth if available
try:
    gp = pd.read_csv(GP_COHORT_CSV, usecols=["race", "ethnicity"])
    race_lower = gp["race"].str.strip().str.lower().fillna("")
    eth_lower  = gp["ethnicity"].str.strip().str.lower().fillna("")
    gp_white   = race_lower.str.contains("white|caucasian", na=False).mean() * 100
    gp_black   = race_lower.str.contains("black|african",   na=False).mean() * 100
    gp_hisp    = eth_lower.str.contains("hispanic|latino|spanish", na=False).mean() * 100

    print("\nGP vs Comparator demographics:")
    print(f"  {'Variable':<22}  {'GP':>8}  {'Comparator':>12}")
    print("  " + "-" * 46)
    print(f"  {'race_white':<22}  {gp_white:>7.1f}%  {out_df['race_white'].mean()*100:>11.1f}%")
    print(f"  {'race_black':<22}  {gp_black:>7.1f}%  {out_df['race_black'].mean()*100:>11.1f}%")
    print(f"  {'ethnicity_hispanic':<22}  {gp_hisp:>7.1f}%  {out_df['ethnicity_hispanic'].mean()*100:>11.1f}%")
except Exception:
    print("  (GP cohort file not available for cross-check)")

assert out_df["patient_id"].is_unique, "Duplicate patient_ids in output"
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV} ({len(out_df):,} rows, {len(out_df.columns)} cols)")
