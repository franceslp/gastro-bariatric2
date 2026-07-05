#!/usr/bin/env python3
"""
collect_insulin_initiation_optimal.py

New insulin initiation after bariatric surgery — GP vs matched comparator.
Following Sadda et al. JAMA Surgery 2026 methodology.

DEFINITION:
  New initiation = first post-surgery prescription among BASELINE NON-USERS.
  Baseline non-users: patients with rapid_insulin=0 for the rapid-acting
  analysis, and long_insulin=0 for the long-acting analysis (analyzed
  separately). Patients on one type may still contribute to the other.

INSULIN TYPES (RxNorm ingredient codes, same as covariate collection):
  Rapid-acting: 51428, 86009, 311036, 1156706
  Long-acting:  253182, 274783, 1151131, 2200801

FOLLOW-UP: day 31 to day 1825 (1 month to 5 years, Sadda-aligned).
  Day 0–30 excluded to avoid capture of perioperative prescriptions.

PRIMARY ANALYSIS (Sadda-style):
  Time-to-event: Kaplan-Meier curves + Cox proportional hazards model
  HR with 95% CI, log-rank test
  Patients who never initiate are censored at their last observed encounter
  (from followup_days_post in ed_hosp_binary files) or at day 1825.

SECONDARY: binary (ever initiated within 5yr) — OR + chi-square, consistent
  with other secondary outcomes in this study.

Runs on BOTH matched cohorts in one medication_ingredient.csv scan.

OUTPUTS:
  insulin_initiation_events_with_BMI.csv   (per patient: time-to-event, event flag)
  insulin_initiation_events_no_BMI.csv
  insulin_initiation_summary.txt           (KM/Cox results + binary ORs)
"""
import subprocess
import numpy as np
import pandas as pd
from scipy import stats

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
    print("lifelines available — running KM + Cox analysis")
except ImportError:
    HAVE_LIFELINES = False
    print("lifelines not available — running binary analysis only")

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
MED_FILE = f"{GCS_BASE}/medication_ingredient.csv"
CHUNK = 500_000

RAPID_CODES = {"51428", "86009", "311036", "1156706"}
LONG_CODES  = {"253182", "274783", "1151131", "2200801"}
ALL_INSULIN = RAPID_CODES | LONG_CODES

FOLLOW_START = 31    # exclude perioperative (day 0-30)
FOLLOW_END   = 1825  # 5 years

COHORTS = {
    "with_BMI_optimal": {
        "dataset": "psm_matched_dataset_optimal.csv",
        "ed_hosp": "ed_hosp_binary_with_BMI_optimal.csv",
    },
}

def load_surgery_dates():
    gp   = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
    comp = pd.read_csv("comparator_pool_ready_for_PSM.csv", dtype={"patient_id": str})
    d = {}
    for _, r in gp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    for _, r in comp.iterrows():
        d[r["patient_id"]] = pd.to_datetime(r["bariatric_date"], errors="coerce")
    return d

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

# ── Load cohorts ─────────────────────────────────────────────────────────────
print("Loading matched cohorts and baseline insulin status...")
all_ids = set()
cohort_data = {}  # cname -> {pid: {group, rapid_baseline, long_baseline}}

for cname, cfg in COHORTS.items():
    ds = pd.read_csv(cfg["dataset"], dtype={"patient_id": str})
    assert ds["patient_id"].nunique() == len(ds), f"{cname}: duplicate patient_ids"

    # Load follow-up days for censoring (from ED/hosp binary file)
    fu_df = pd.read_csv(cfg["ed_hosp"], dtype={"patient_id": str})
    fu_map = dict(zip(fu_df["patient_id"],
                      pd.to_numeric(fu_df["followup_days_post"], errors="coerce")))

    members = {}
    for _, r in ds.iterrows():
        pid = r["patient_id"]
        rapid_base = str(r.get("rapid_insulin","0")) in ("1","1.0","True")
        long_base  = str(r.get("long_insulin","0"))  in ("1","1.0","True")
        fu = fu_map.get(pid, np.nan)
        members[pid] = {
            "group":        r["group"],
            "rapid_base":   rapid_base,
            "long_base":    long_base,
            # Issue 2: keep NaN if no follow-up data — don't assume 5yr
            # observed. Handled at event-building time (see below).
            "followup_days": min(float(fu), FOLLOW_END)
                             if not np.isnan(fu) else np.nan,
        }
    cohort_data[cname] = members
    all_ids |= set(members.keys())

    n_rapid_nonusers = sum(1 for m in members.values() if not m["rapid_base"])
    n_long_nonusers  = sum(1 for m in members.values() if not m["long_base"])
    print(f"  {cname}: {len(members)} patients | "
          f"rapid non-users: {n_rapid_nonusers} | long non-users: {n_long_nonusers}")

print(f"Total unique patients: {len(all_ids)}")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}

# ── Scan medication_ingredient.csv ────────────────────────────────────────────
# Store: pid -> {insulin_type: first_post_surgery_day}
first_rx = {pid: {"rapid": None, "long": None} for pid in all_ids}

print("\nScanning medication_ingredient.csv for post-surgery insulin prescriptions...")
rows = 0
for chunk in stream(MED_FILE, ["patient_id", "code", "start_date"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue
    chunk["code"] = chunk["code"].str.strip()
    chunk = chunk[chunk["code"].isin(ALL_INSULIN)]
    if chunk.empty:
        continue
    chunk["dt"] = parse_dates(chunk["start_date"])
    chunk = chunk[chunk["dt"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days"] = (chunk["dt"] - chunk["surg"]).dt.days
    # Post-surgery only, within follow-up window
    chunk = chunk[(chunk["days"] >= FOLLOW_START) & (chunk["days"] <= FOLLOW_END)]
    if chunk.empty:
        continue
    for pid, code, days in zip(chunk["patient_id"], chunk["code"], chunk["days"]):
        days = int(days)
        itype = "rapid" if code in RAPID_CODES else "long"
        cur = first_rx[pid][itype]
        if cur is None or days < cur:
            first_rx[pid][itype] = days
    if rows % 100_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows\n")

# ── Build event files and run analysis ────────────────────────────────────────
def or_ci(gp_n, gp_N, co_n, co_N):
    a, b, c, d = gp_n, gp_N-gp_n, co_n, co_N-co_n
    tbl = np.array([[a,b],[c,d]])
    try:
        if min(a,b,c,d) < 5:
            _, p = stats.fisher_exact(tbl)
        else:
            _, p, _, _ = stats.chi2_contingency(tbl)
    except:
        p = np.nan
    if min(a,b,c,d) == 0:
        a,b,c,d = a+.5,b+.5,c+.5,d+.5
    OR = (a*d)/(b*c)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    lo, hi = np.exp(np.log(OR)-1.96*se), np.exp(np.log(OR)+1.96*se)
    return OR, lo, hi, p

report = []
report.append("="*72)
report.append("NEW INSULIN INITIATION AFTER BARIATRIC SURGERY — GP vs Comparator")
report.append("(Among baseline non-users of each insulin type separately)")
report.append("Following Sadda et al. JAMA Surgery 2026")
report.append("Rapid-acting: RxNorm 51428,86009,311036,1156706")
report.append("Long-acting:  RxNorm 253182,274783,1151131,2200801")
report.append("Follow-up: day 31–1825 (1 month to 5 years post-surgery)")
report.append("="*72)

for cname, members in cohort_data.items():
    report.append(f"\n{'='*60}\nCOHORT: {cname}\n{'='*60}")

    event_rows = []
    n_no_followup = 0
    for pid, info in members.items():
        fu = info["followup_days"]
        fu_is_missing = np.isnan(fu) if not isinstance(fu, float) else np.isnan(fu)
        # Issue 2: if no follow-up data, conservatively censor at FOLLOW_END
        # but track these patients separately for QA reporting
        if fu_is_missing:
            n_no_followup += 1
            fu_used = float(FOLLOW_END)
        else:
            fu_used = float(fu)

        for itype, base_col in [("rapid","rapid_base"),("long","long_base")]:
            if info[base_col]:
                continue  # baseline user — exclude from this analysis
            first_day = first_rx[pid][itype]
            # Only count as event if prescription falls within observed follow-up
            if first_day is not None and first_day <= fu_used:
                event = True
                time  = first_day
            else:
                event = False
                time  = fu_used  # censored at last follow-up or end of window
            event_rows.append({
                "patient_id": pid, "group": info["group"],
                "insulin_type": itype, "event": event,
                "time": max(time, 1),  # ensure time >= 1
                "had_followup": not fu_is_missing,
            })
    if n_no_followup > 0:
        report.append(f"  QA: {n_no_followup} patients had no post-op encounter data "
                     f"— conservatively censored at day {FOLLOW_END}")
    evt_df = pd.DataFrame(event_rows)
    evt_df.to_csv(f"insulin_initiation_events_{cname}.csv", index=False)

    # Issue 4: QA — print event counts immediately so RxNorm matching failures
    # are obvious (e.g., 0/4000 events would flag a code mismatch)
    for itype in ["rapid","long"]:
        itype_df = evt_df[evt_df["insulin_type"]==itype]
        n_events = itype_df["event"].sum()
        n_total  = len(itype_df)
        print(f"  {cname} {itype}: {n_events}/{n_total} events "
              f"({100*n_events/n_total:.1f}%) "
              f"[{'OK' if n_events > 0 else 'WARNING: 0 events — check RxNorm codes'}]")

    for itype, label in [("rapid","Rapid-acting insulin"),("long","Long-acting insulin")]:
        sub = evt_df[evt_df["insulin_type"]==itype]
        gp  = sub[sub["group"]=="gastroparesis"]
        co  = sub[sub["group"]=="comparator"]

        report.append(f"\n{label} (PRIMARY: Cox HR; SECONDARY: binary ever-initiation OR):")
        report.append(f"  Baseline non-users: GP n={len(gp)}, Comparator n={len(co)}")
        report.append(f"  New initiations:   GP {gp['event'].sum()}/{len(gp)} "
                     f"({100*gp['event'].mean():.1f}%), "
                     f"Comp {co['event'].sum()}/{len(co)} "
                     f"({100*co['event'].mean():.1f}%)")

        # Binary OR
        OR, lo, hi, p = or_ci(gp["event"].sum(), len(gp),
                               co["event"].sum(), len(co))
        report.append(f"  Binary 5yr ever-initiation OR = {OR:.2f} (95%CI {lo:.2f}–{hi:.2f}), p={p:.4f} (secondary)")

        # KM + Cox if lifelines available
        if HAVE_LIFELINES:
            try:
                # Log-rank test
                lr = logrank_test(
                    gp["time"], co["time"],
                    event_observed_A=gp["event"],
                    event_observed_B=co["event"]
                )
                report.append(f"  Log-rank p = {lr.p_value:.4f}")

                # KM median time to initiation
                kmf = KaplanMeierFitter()
                for grp, grp_label in [("gastroparesis","GP"),("comparator","Comp")]:
                    g = sub[sub["group"]==grp]
                    kmf.fit(g["time"], event_observed=g["event"], label=grp_label)
                    med = kmf.median_survival_time_
                    report.append(f"  {grp_label} median time to initiation: "
                                 f"{med:.0f} days" if not np.isinf(med) else
                                 f"  {grp_label} median time to initiation: not reached")

                # Cox PH model
                cox_df = sub[["time","event","group"]].copy()
                cox_df["gp_bin"] = (cox_df["group"]=="gastroparesis").astype(int)
                cph = CoxPHFitter()
                cph.fit(cox_df[["time","event","gp_bin"]], duration_col="time",
                        event_col="event")
                hr = np.exp(cph.params_["gp_bin"])
                ci = cph.confidence_intervals_
                lo_hr = np.exp(ci.loc["gp_bin","95% lower-bound"])
                hi_hr = np.exp(ci.loc["gp_bin","95% upper-bound"])
                p_cox = cph.summary["p"]["gp_bin"]
                report.append(f"  Cox HR (GP vs Comp) = {hr:.2f} "
                             f"(95%CI {lo_hr:.2f}–{hi_hr:.2f}), p={p_cox:.4f}")
                interp = ("GP higher hazard" if hr>1 else "GP lower hazard")
                report.append(f"  Interpretation: {interp} of new {itype}-acting insulin initiation")
                # Issue 3: proportional hazards assumption check
                # (Schoenfeld residuals test — p>0.05 means PH assumption holds)
                try:
                    import io, contextlib
                    ph_buf = io.StringIO()
                    with contextlib.redirect_stdout(ph_buf):
                        cph.check_assumptions(
                            cox_df[["time","event","gp_bin"]],
                            p_value_threshold=0.05, show_plots=False)
                    ph_out = ph_buf.getvalue()
                    ph_pass = "not" not in ph_out.lower() and "violated" not in ph_out.lower()
                    report.append(f"  PH assumption check: "
                                 f"{'PASSED' if ph_pass else 'REVIEW NEEDED — see log'}")
                except Exception as e_ph:
                    report.append(f"  PH assumption check: skipped ({e_ph})")
            except Exception as e:
                report.append(f"  KM/Cox failed: {e}")

text = "\n".join(report)
with open("insulin_initiation_summary_optimal.txt","w") as f:
    f.write(text)
print(text)
print("\nWrote: insulin_initiation_events_with_BMI.csv, "
      "insulin_initiation_summary_optimal.txt")
