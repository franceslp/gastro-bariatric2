#!/usr/bin/env python3
"""
collect_ed_hospitalizations.py

ED visits and hospitalizations for matched GP-bariatric vs comparator cohorts,
with PRE/POST surgery comparison and stay-level deduplication.

Following Sadda et al. JAMA Surgery 2026 (post-op window) plus a symmetric
pre-operative window for within-patient pre/post comparison.

DEFINITIONS:
  - ED visit:        encounter type = 'EMER'
  - Hospitalization: encounter type = 'IMP' (inpatient)

DEDUPLICATION (important — TriNetX codes one inpatient stay as many rows):
  Encounters of the same type within GAP_DAYS=1 of each other (same or
  consecutive days) are collapsed into ONE distinct event ("stay"/"visit").
  A new event starts only after a gap of 2+ days. This removes per-day /
  per-service coding artifacts while preserving genuine readmissions.
  Applied to BOTH inpatient and ED.

WINDOWS:
  - POST-surgery: day 31 to day 1825 (1 month to 5 years; Sadda-aligned).
    Events within the first 30 postoperative days are EXCLUDED to avoid
    capturing perioperative/index-admission events (Sadda methodology).
  - PRE-surgery:  day -1825 to day -1 (5 years before; ~symmetric, 1825 vs
    1795 days — the 30-day post exclusion makes post slightly shorter; this
    asymmetry is stated rather than adjusted, per Sadda alignment).
  Distinct-event counts computed in each, plus annual post-op breakdown.

PRIMARY comparisons:
  - Binary 5yr (ever/never) GP vs comparator, post-op (Sadda-style), OR + chi2
  - Pre vs post WITHIN each group (paired): did event rate change after surgery?
  - Between-group difference-in-differences (post-pre), GP vs comparator

Runs on BOTH matched cohorts in one encounter.csv scan.

Encounter type codes confirmed from dataset: EMER, IMP.

OUTPUTS:
  ed_hosp_binary_with_BMI.csv / _no_BMI.csv   (per patient: pre/post distinct counts + binary)
  ed_hosp_annual_with_BMI.csv / _no_BMI.csv   (annual post-op, long format)
  ed_hosp_summary.txt
"""
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

GCS_BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
ENCOUNTER_FILE = f"{GCS_BASE}/encounter.csv"
CHUNK = 500_000

ED_TYPE, IP_TYPE = "EMER", "IMP"
GAP_DAYS = 1  # encounters within 1 day = same distinct event

POST_START, POST_END = 31, 1825      # 1 month to 5 years after
PRE_START,  PRE_END  = -1825, -1     # 5 years before

ANNUAL_WINDOWS = {
    "year_1": (31,   365),  "year_2": (366,  730), "year_3": (731,  1095),
    "year_4": (1096, 1460), "year_5": (1461, 1825),
}

COHORTS = {
    "with_BMI": {"dataset": "psm_matched_dataset_new.csv"},
    "no_BMI":   {"dataset": "psm_matched_dataset_no_BMI.csv"},
}

def load_surgery_dates():
    gp = pd.read_csv("cohort_FINAL_analytic.csv", dtype={"patient_id": str})
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

def dedup_events(days_list, gap=GAP_DAYS):
    """Collapse a sorted list of day-offsets into distinct events.
    Encounters within `gap` days of the previous one are the same event."""
    if not days_list:
        return []
    s = sorted(days_list)
    events = [s[0]]          # keep the start day of each distinct event
    last = s[0]
    for d in s[1:]:
        if d - last > gap:   # gap exceeded -> new distinct event
            events.append(d)
        last = d
    return events            # list of event start-days

def count_in_window(event_days, d0, d1):
    return sum(1 for d in event_days if d0 <= d <= d1)

print("Loading matched cohorts...")
all_ids = set()
cohort_members = {}
for cname, cfg in COHORTS.items():
    ds = pd.read_csv(cfg["dataset"], dtype={"patient_id": str})
    assert ds["patient_id"].nunique() == len(ds), f"{cname}: duplicate patient_ids"
    members = {r["patient_id"]: r["group"] for _, r in ds.iterrows()}
    cohort_members[cname] = members
    all_ids |= set(members.keys())
    print(f"  {cname}: {len(members)} matched patients")
print(f"Total unique patients: {len(all_ids)}")

surgery_dates = load_surgery_dates()
surgery_dt = {pid: surgery_dates.get(pid, pd.NaT) for pid in all_ids}

# Raw day-offsets per patient per type (full range; dedup applied later)
raw = {pid: {ED_TYPE: [], IP_TYPE: []} for pid in all_ids}
# Issue 6: track last observed encounter day per patient (ANY type) to estimate
# post-surgery follow-up time (loss-to-follow-up / denominator transparency)
last_obs_day = {pid: None for pid in all_ids}

print("\nScanning encounter.csv for EMER and IMP (full pre+post range)...")
rows = 0
type_counts_seen = {}
for chunk in stream(ENCOUNTER_FILE, ["patient_id", "start_date", "type"]):
    rows += len(chunk)
    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue
    chunk["type"] = chunk["type"].str.strip()
    for t, n in chunk["type"].value_counts().items():
        type_counts_seen[t] = type_counts_seen.get(t, 0) + n
    chunk["dt"] = parse_dates(chunk["start_date"])
    chunk = chunk[chunk["dt"].notna()]
    chunk["surg"] = chunk["patient_id"].map(surgery_dt)
    chunk = chunk[chunk["surg"].notna()]
    chunk["days"] = (chunk["dt"] - chunk["surg"]).dt.days
    # Issue 6: update last observed POST-surgery day from ANY encounter type
    post_any = chunk[chunk["days"] >= 1]
    for pid, days in zip(post_any["patient_id"], post_any["days"]):
        prev = last_obs_day[pid]
        if prev is None or days > prev:
            last_obs_day[pid] = int(days)
    # Now restrict to EMER/IMP within the +/-5yr range for event counting
    chunk = chunk[chunk["type"].isin([ED_TYPE, IP_TYPE])]
    chunk = chunk[(chunk["days"] >= PRE_START) & (chunk["days"] <= POST_END)]
    if chunk.empty:
        continue
    for pid, enc_type, days in zip(chunk["patient_id"], chunk["type"], chunk["days"]):
        raw[pid][enc_type].append(int(days))
    if rows % 50_000_000 == 0:
        print(f"  ...{rows:,} rows scanned")

print(f"Done scanning: {rows:,} rows")
print(f"Encounter types seen (matched patients): {type_counts_seen}")

# Dedup into distinct events per patient per type
events = {pid: {ED_TYPE: dedup_events(raw[pid][ED_TYPE]),
                IP_TYPE: dedup_events(raw[pid][IP_TYPE])}
          for pid in all_ids}

# Report raw-vs-dedup compression (QA: shows the per-day coding effect)
raw_ip = sum(len(raw[p][IP_TYPE]) for p in all_ids)
ded_ip = sum(len(events[p][IP_TYPE]) for p in all_ids)
raw_ed = sum(len(raw[p][ED_TYPE]) for p in all_ids)
ded_ed = sum(len(events[p][ED_TYPE]) for p in all_ids)
print(f"\nDeduplication (gap={GAP_DAYS}d):")
print(f"  Inpatient rows {raw_ip} -> {ded_ip} distinct stays "
      f"({100*(1-ded_ip/raw_ip):.0f}% were same-stay duplicates)" if raw_ip else "  no IP")
print(f"  ED rows {raw_ed} -> {ded_ed} distinct visits "
      f"({100*(1-ded_ed/raw_ed):.0f}% were same-visit duplicates)" if raw_ed else "  no ED")

report = []
report.append("="*68)
report.append("ED VISITS & HOSPITALIZATIONS — GP vs Comparator (pre/post surgery)")
report.append(f"Distinct events via {GAP_DAYS}-day gap dedup. "
              f"Post: d31-1825; Pre: d-1825 to -1.")
report.append("="*68)
report.append(f"Dedup: IP {raw_ip}->{ded_ip} stays, ED {raw_ed}->{ded_ed} visits")

for cname, cfg in COHORTS.items():
    members = cohort_members[cname]
    report.append(f"\n{'='*58}\nCOHORT: {cname}\n{'='*58}")

    rows_out = []
    for pid, grp in members.items():
        ed_post  = count_in_window(events[pid][ED_TYPE], POST_START, POST_END)
        ed_pre   = count_in_window(events[pid][ED_TYPE], PRE_START,  PRE_END)
        ip_post  = count_in_window(events[pid][IP_TYPE], POST_START, POST_END)
        ip_pre   = count_in_window(events[pid][IP_TYPE], PRE_START,  PRE_END)
        acute_post = ed_post + ip_post
        acute_pre  = ed_pre + ip_pre
        rows_out.append({
            "patient_id": pid, "group": grp,
            "ED_pre_5yr": ed_pre,  "ED_post_5yr": ed_post,
            "IP_pre_5yr": ip_pre,  "IP_post_5yr": ip_post,
            "acute_pre_5yr": acute_pre, "acute_post_5yr": acute_post,
            "had_ED_post": ed_post > 0, "had_ED_pre": ed_pre > 0,
            "had_IP_post": ip_post > 0, "had_IP_pre": ip_pre > 0,
            "had_acute_post": acute_post > 0, "had_acute_pre": acute_pre > 0,
            "ED_change": ed_post - ed_pre, "IP_change": ip_post - ip_pre,
            "acute_change": acute_post - acute_pre,
            "followup_days_post": last_obs_day.get(pid),  # Issue 6
        })
    bdf = pd.DataFrame(rows_out)
    bdf.to_csv(f"ed_hosp_binary_{cname}.csv", index=False)

    # Annual post-op long format
    annual = []
    for pid, grp in members.items():
        for outcome, et in [("ED", ED_TYPE), ("hospitalization", IP_TYPE)]:
            for yr, (d0, d1) in ANNUAL_WINDOWS.items():
                n = count_in_window(events[pid][et], d0, d1)
                annual.append({"patient_id": pid, "group": grp, "outcome": outcome,
                               "year": yr, "count": n, "had_visit": n > 0})
    pd.DataFrame(annual).to_csv(f"ed_hosp_annual_{cname}.csv", index=False)

    def or_ci(gp_n, gp_N, co_n, co_N):
        a, b, c, d = gp_n, gp_N-gp_n, co_n, co_N-co_n
        tbl = np.array([[a, b],[c, d]])
        try:
            _, p, _, _ = stats.chi2_contingency(tbl)
        except Exception:
            p = np.nan
        # Haldane-Anscombe correction: if any cell is 0, add 0.5 to all cells
        # so the OR/CI remain defined (rare-event outcomes don't disappear).
        if min(a,b,c,d) == 0:
            a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
        OR = (a*d)/(b*c); se = np.sqrt(1/a+1/b+1/c+1/d)
        lo, hi = np.exp(np.log(OR)-1.96*se), np.exp(np.log(OR)+1.96*se)
        return OR, lo, hi, p

    # ---- POST-op binary GP vs comparator (Sadda-style) ----
    for col, lab in [("had_ED_post","ED visit"), ("had_IP_post","Hospitalization"),
                     ("had_acute_post","Acute care (ED or hospitalization)")]:
        gp = bdf[bdf["group"]=="gastroparesis"][col]
        co = bdf[bdf["group"]=="comparator"][col]
        OR, lo, hi, p = or_ci(gp.sum(), len(gp), co.sum(), len(co))
        report.append(f"\n{lab} — POST-op 5yr (binary, GP vs comparator):")
        report.append(f"  GP {gp.sum()}/{len(gp)} ({100*gp.mean():.1f}%) vs "
                     f"Comp {co.sum()}/{len(co)} ({100*co.mean():.1f}%)")
        report.append(f"  OR={OR:.2f} (95%CI {lo:.2f}-{hi:.2f}), p={p:.4f}")

    # ---- PRE vs POST within each group (paired) ----
    for et_col_pre, et_col_post, lab in [
        ("ED_pre_5yr","ED_post_5yr","ED visits"),
        ("IP_pre_5yr","IP_post_5yr","Hospitalizations"),
        ("acute_pre_5yr","acute_post_5yr","Acute care events")]:
        report.append(f"\n{lab} — PRE vs POST (within-group, distinct-event counts):")
        for grp in ["gastroparesis","comparator"]:
            g = bdf[bdf["group"]==grp]
            pre, post = g[et_col_pre], g[et_col_post]
            # paired Wilcoxon (counts, non-normal)
            try:
                w, pw = stats.wilcoxon(post, pre, zero_method="wilcox")
            except Exception:
                pw = np.nan
            report.append(f"  {grp}: pre mean {pre.mean():.2f} -> post mean "
                         f"{post.mean():.2f} (Δ={post.mean()-pre.mean():+.2f}, "
                         f"Wilcoxon p={pw:.4f})")

    # ---- Difference-in-differences (post-pre), GP vs comparator ----
    for change_col, lab in [("ED_change","ED visits"),("IP_change","Hospitalizations"),
                            ("acute_change","Acute care events")]:
        gp = bdf[bdf["group"]=="gastroparesis"][change_col]
        co = bdf[bdf["group"]=="comparator"][change_col]
        try:
            t, p = stats.ttest_ind(gp, co, equal_var=False)
        except Exception:
            p = np.nan
        report.append(f"\n{lab} — change (post-pre) GP vs comparator (diff-in-diff):")
        report.append(f"  GP Δ mean {gp.mean():+.2f} vs Comp Δ mean {co.mean():+.2f}, "
                     f"p={p:.4f}")

    # ---- Issue 4: negative binomial regression on post-op counts ----
    # Count outcomes (n events) are over-dispersed with many zeros; NB regression
    # is the appropriate model. post_count ~ group + pre_count adjusts for each
    # patient's own baseline utilization rate (stronger than unadjusted comparison).
    if HAVE_SM:
        report.append(f"\nNegative binomial regression (post counts ~ group + pre count):")
        for pre_col, post_col, lab in [
            ("ED_pre_5yr","ED_post_5yr","ED visits"),
            ("IP_pre_5yr","IP_post_5yr","Hospitalizations"),
            ("acute_pre_5yr","acute_post_5yr","Acute care events")]:
            md = bdf[["group", pre_col, post_col, "followup_days_post"]].copy()
            md["gp_bin"] = (md["group"]=="gastroparesis").astype(int)
            md = md.rename(columns={pre_col:"pre_count", post_col:"post_count"})
            # Issue 1: log(follow-up years) offset converts model to rates per
            # person-year. Without this, patients with short follow-up contribute
            # the same denominator as those with 5 years (inflates apparent rates).
            md["fu_days"] = pd.to_numeric(md["followup_days_post"], errors="coerce")
            md["fu_days"] = md["fu_days"].clip(lower=1)  # avoid log(0)
            md["log_fu"] = np.log(md["fu_days"] / 365)
            md = md.dropna(subset=["fu_days"])
            try:
                try:
                    m = smf.glm("post_count ~ gp_bin + pre_count", data=md,
                                family=sm.families.NegativeBinomial(),
                                offset=md["log_fu"]).fit()
                    fam = "NB"
                except Exception:
                    m = smf.glm("post_count ~ gp_bin + pre_count", data=md,
                                family=sm.families.Poisson(),
                                offset=md["log_fu"]).fit()
                    fam = "Poisson"
                # Issue 3: with offset this is a true incidence rate ratio
                irr = np.exp(m.params.get("gp_bin", np.nan))
                ci = m.conf_int()
                lo = np.exp(ci.loc["gp_bin",0]) if "gp_bin" in ci.index else np.nan
                hi = np.exp(ci.loc["gp_bin",1]) if "gp_bin" in ci.index else np.nan
                p  = m.pvalues.get("gp_bin", np.nan)
                report.append(f"  {lab} [{fam}, offset=log(follow-up years)]: "
                             f"GP incidence rate ratio = "
                             f"{irr:.2f} (95%CI {lo:.2f}-{hi:.2f}), p={p:.4f}")
            except Exception as e:
                report.append(f"  {lab}: regression failed ({e})")

    # Issue 6: follow-up time summary (loss-to-follow-up transparency)
    report.append(f"\nObserved post-surgery follow-up (last encounter, any type):")
    for grp in ["gastroparesis", "comparator"]:
        fu_raw = pd.to_numeric(
            bdf[bdf["group"]==grp]["followup_days_post"], errors="coerce")
        missing_fu = int(fu_raw.isna().sum())  # capture BEFORE dropna
        fu = fu_raw.dropna()
        if len(fu):
            report.append(f"  {grp}: median {fu.median():.0f} days "
                         f"({fu.median()/365:.1f} yr), "
                         f"{int((fu>=1825).sum())}/{len(fu)} reached 5yr, "
                         f"{missing_fu} no post-op encounter")
    print(f"{cname}: wrote ed_hosp_binary_{cname}.csv, ed_hosp_annual_{cname}.csv")

text = "\n".join(report)
with open("ed_hosp_summary.txt","w") as f:
    f.write(text)
print(text)
print("\nWrote: ed_hosp_summary.txt")
