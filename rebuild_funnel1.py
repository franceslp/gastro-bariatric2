#!/usr/bin/env python3
"""
rebuild_funnel1.py  --  corrected anchors + multi-surgery flags + exact format
                        + aggressive "would-have-matched-if" parsing audit.

PRIMARY anchor = earliest of {43644,43775}; SENSITIVITY anchor = earliest of all five.
Flags (not exclusions): multi_surgery, same_day_ambiguous, unlisted (43659/43999).
Output matches Master Cohort columns; demographic/medication cols left blank.

PARSING AUDIT: beyond the strict normalization used for the actual cohort, the
script also counts, per code type, how many EXTRA unique patients each looser
matching rule would add (modifiers, internal whitespace, trailing-zero ICD,
unanticipated suffixes). These are REPORTED ONLY -- they do not enter the cohort
unless you decide to accept them. Read these sections before trusting counts.
"""

import subprocess, re
import pandas as pd
from collections import defaultdict

BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAGNOSIS_URI = f"{BASE}/diagnosis.csv"
PROCEDURE_URI = f"{BASE}/procedure.csv"
OCT2015 = pd.Timestamp("2015-10-01")

PRIMARY    = {"43644", "43775"}
SENS_EXTRA = {"43645", "43846", "43847"}
ALLFIVE    = PRIMARY | SENS_EXTRA
UNLISTED   = {"43659", "43999"}
GES_CODES  = {"78264", "78265", "78266"}

MASTER_COLS = ['patient_id','year_of_birth','age_at_surgery_approx','bariatric_date',
 'bariatric_cpt_codes_seen','closest_K31_84_strictly_before_surgery',
 'num_K31_84_encounters','K3184_span_days','all_K31_84_dates','diabetes_type_label',
 'closest_E10_E11_code_before_surgery','closest_E10_E11_date_before_surgery',
 'closest_GES_before_K3184_dx_code','closest_GES_before_K3184_dx_date',
 'days_GES_before_K3184','first_prokinetic_after_K3184_dx_drug',
 'first_prokinetic_after_K3184_dx_date','days_to_prokinetic_after_K3184','sex','race',
 'ethnicity','marital_status','month_year_death','deceased','meets_age_requirement']

# ---------- normalization tiers ----------
def cpt_strict(v):
    s = str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s

def cpt_loose(v):
    """strip whitespace anywhere, drop a modifier after '-', strip trailing .0"""
    s = re.sub(r"\s+", "", str(v).upper())
    s = s.split("-")[0]                     # drop CPT modifier
    if s.endswith(".0"): s = s[:-2]
    return s

def icd_strict(v):
    return re.sub(r"\.", "", str(v).strip().upper())

def icd_loose(v):
    """remove all whitespace + dots; collapse a trailing zero on K3184x"""
    s = re.sub(r"[\s.]", "", str(v).upper())
    return s

def find_col(cols, c, k):
    lo = [x.lower() for x in cols]
    for cand in c:
        if cand in lo: return cols[lo.index(cand)]
    for i, x in enumerate(lo):
        if k in x: return cols[i]
    return None

def scan_procedure():
    print("PASS 1: PROCEDURE", flush=True)
    p = subprocess.Popen(["gsutil","cat",PROCEDURE_URI], stdout=subprocess.PIPE)
    surg = defaultdict(list); unlisted = set(); ges = defaultdict(list)
    raw = defaultdict(int); cs = defaultdict(int)
    # parsing audit: pids matched per tier
    strict_pts = defaultdict(set); loose_pts = defaultdict(set)
    audit_examples = defaultdict(lambda: defaultdict(int))
    rows = 0; cc = dc = cscol = None
    targets = ALLFIVE | UNLISTED | GES_CODES
    stems = {c[:4] for c in targets}
    for ch in pd.read_csv(p.stdout, dtype=str, chunksize=500_000):
        rows += len(ch)
        if cc is None:
            cc = find_col(ch.columns, ["code","cpt_code","procedure_code","code_value"], "code")
            dc = find_col(ch.columns, ["date","procedure_date","service_date","date_of_service","start_date"], "date")
            cscol = next((c for c in ch.columns if "system" in c.lower()), None)
            print(f"code='{cc}' date='{dc}' system='{cscol}'", flush=True)
            print(f"columns: {list(ch.columns)}", flush=True)
        ch["_s"] = ch[cc].map(cpt_strict)
        ch["_l"] = ch[cc].map(cpt_loose)

        # actual cohort uses STRICT on the five
        m = ch[ch["_s"].isin(ALLFIVE)]
        if not m.empty:
            if cscol:
                for v,n in m[cscol].value_counts(dropna=False).items(): cs[str(v)]+=int(n)
            for v in m[cc]: raw[str(v)]+=1
            d = pd.to_datetime(m[dc], errors="coerce")
            for pid,c,dt in zip(m["patient_id"], m["_s"], d):
                if pd.notna(dt): surg[pid].append((dt,c))
        for pid in ch[ch["_s"].isin(UNLISTED)]["patient_id"]: unlisted.add(pid)
        g = ch[ch["_s"].isin(GES_CODES)]
        if not g.empty:
            d = pd.to_datetime(g[dc], errors="coerce")
            for pid,c,dt in zip(g["patient_id"], g["_s"], d):
                if pd.notna(dt): ges[pid].append((dt,c))

        # PARSING AUDIT for the five bariatric codes
        for code in ALLFIVE:
            s_hit = ch[ch["_s"]==code]
            strict_pts[code].update(s_hit["patient_id"])
            l_hit = ch[(ch["_l"]==code) & (ch["_s"]!=code)]   # loose-only matches
            loose_pts[code].update(l_hit["patient_id"])
            for v in l_hit[cc]:
                audit_examples[code][str(v)] += 1

        if rows % 50_000_000 == 0: print(f"  ...{rows:,} rows, {len(surg):,} pts", flush=True)
    p.stdout.close(); p.wait()
    print(f"Procedure rows: {rows:,}; bariatric pts (strict): {len(surg):,}; unlisted: {len(unlisted):,}")
    print(f"code_system: {dict(cs)}")
    print("raw matched bariatric values (strict):", dict(sorted(raw.items())))
    print("\n--- CPT PARSING AUDIT (extra patients if looser matching accepted) ---")
    for code in sorted(ALLFIVE):
        extra = loose_pts[code] - strict_pts[code]
        if extra:
            print(f"  {code}: +{len(extra)} extra patients via loose match. Raw values: {dict(audit_examples[code])}")
        else:
            print(f"  {code}: no additional patients from looser matching (clean)")
    return surg, unlisted, ges

def scan_diagnosis(prim_date, sens_date):
    print("PASS 2: DIAGNOSIS", flush=True)
    ids = set(prim_date) | set(sens_date)
    p = subprocess.Popen(["gsutil","cat",DIAGNOSIS_URI], stdout=subprocess.PIPE)
    kdates = defaultdict(list); ecode = defaultdict(list)
    rawk = defaultdict(int)
    k_strict = set(); k_loose = set(); k_audit = defaultdict(int)
    e_strict = set(); e_loose = set(); e_audit = defaultdict(int)
    rows = 0; cc = dc = None
    for ch in pd.read_csv(p.stdout, dtype=str, chunksize=500_000):
        rows += len(ch)
        if cc is None:
            cc = find_col(ch.columns, ["code","icd_code","diagnosis_code","dx_code","code_value"], "code")
            dc = find_col(ch.columns, ["date","diagnosis_date","service_date","date_of_service","start_date"], "date")
            print(f"code='{cc}' date='{dc}'", flush=True)
            print(f"columns: {list(ch.columns)}", flush=True)
        sub = ch[ch["patient_id"].isin(ids)]
        if sub.empty:
            if rows % 50_000_000 == 0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub = sub.copy()
        sub["_s"] = sub[cc].map(icd_strict)
        sub["_l"] = sub[cc].map(icd_loose)
        sub["_d"] = pd.to_datetime(sub[dc], errors="coerce")

        # actual cohort uses STRICT
        k = sub[(sub["_s"]=="K3184") & (sub["_d"]>=OCT2015)]
        for pid,dt in zip(k["patient_id"], k["_d"]):
            if pd.notna(dt): kdates[pid].append(dt)
        for v in k[cc]: rawk[str(v)]+=1
        e = sub[sub["_s"].str.startswith(("E10","E11"), na=False)]
        for pid,dt,code in zip(e["patient_id"], e["_d"], e[cc]):
            if pd.notna(dt): ecode[pid].append((dt,code))

        # PARSING AUDIT
        k_strict.update(sub[sub["_s"]=="K3184"]["patient_id"])
        kl = sub[(sub["_l"].str.startswith("K3184", na=False)) & (sub["_s"]!="K3184")]
        k_loose.update(kl["patient_id"])
        for v in kl[cc]: k_audit[str(v)] += 1
        e_strict.update(sub[sub["_s"].str.startswith(("E10","E11"), na=False)]["patient_id"])
        el = sub[(sub["_l"].str.startswith(("E10","E11"), na=False)) &
                 (~sub["_s"].str.startswith(("E10","E11"), na=False))]
        e_loose.update(el["patient_id"])
        for v in el[cc]: e_audit[str(v)] += 1

        if rows % 50_000_000 == 0: print(f"  ...{rows:,} rows", flush=True)
    p.stdout.close(); p.wait()
    print(f"Diagnosis rows: {rows:,}")
    print("raw matched K31.84 (strict):", dict(rawk))
    print("\n--- ICD PARSING AUDIT (extra patients if looser matching accepted) ---")
    ke = k_loose - k_strict
    print(f"  K31.84: +{len(ke)} extra via loose match. Raw values: {dict(k_audit)}" if ke
          else "  K31.84: no additional patients from looser matching (clean)")
    ee = e_loose - e_strict
    print(f"  E10/E11: +{len(ee)} extra via loose match. Raw values: {dict(list(e_audit.items())[:30])}" if ee
          else "  E10/E11: no additional patients from looser matching (clean)")
    return kdates, ecode

def main():
    surg, unlisted, ges = scan_procedure()
    prim_date, sens_date, surg_dates_all = {}, {}, {}
    for pid, lst in surg.items():
        prim = [d for d,c in lst if c in PRIMARY]
        if prim: prim_date[pid] = min(prim)
        sens_date[pid] = min(d for d,c in lst)
        surg_dates_all[pid] = sorted(set(d for d,c in lst))
    kdates, ecode = scan_diagnosis(prim_date, sens_date)

    def build(anchor):
        out = []
        for pid, bdate in anchor.items():
            kq = [d for d in kdates.get(pid,[]) if d < bdate]
            eq = [(d,c) for d,c in ecode.get(pid,[]) if d < bdate]
            if not kq or not eq: continue
            codes = sorted(set(c for d,c in surg[pid]))
            dates = surg_dates_all[pid]
            same_day = any(len(set(c for d,c in surg[pid] if d==dd))>1 for dd in dates)
            row = {c:"" for c in MASTER_COLS}
            row.update({
                "patient_id": pid, "bariatric_date": bdate.date(),
                "bariatric_cpt_codes_seen": ",".join(codes),
                "closest_K31_84_strictly_before_surgery": max(kq).date(),
                "num_K31_84_encounters": len(kdates.get(pid,[])),
                "K3184_span_days": (max(kdates[pid])-min(kdates[pid])).days if kdates.get(pid) else "",
                "all_K31_84_dates": ",".join(str(d.date()) for d in sorted(kdates.get(pid,[]))),
                "closest_E10_E11_code_before_surgery": max(eq,key=lambda x:x[0])[1],
                "closest_E10_E11_date_before_surgery": max(eq,key=lambda x:x[0])[0].date(),
            })
            if ges.get(pid):
                gb = [(d,c) for d,c in ges[pid] if d < bdate]
                if gb:
                    gd,gc = max(gb,key=lambda x:x[0])
                    row["closest_GES_before_K3184_dx_code"]=gc
                    row["closest_GES_before_K3184_dx_date"]=gd.date()
            row["_multi_surgery_flag"]=len(dates)>1
            row["_all_bariatric_surgery_dates"]=",".join(str(d.date()) for d in dates)
            row["_second_bariatric_surgery_date"]=str(dates[1].date()) if len(dates)>1 else ""
            row["_same_day_ambiguous"]=same_day
            row["_unlisted_flag"]=pid in unlisted
            out.append(row)
        return pd.DataFrame(out)

    sens = build(sens_date); prim = build(prim_date)
    prim_ids = set(prim["patient_id"])
    sens["_in_primary"] = sens["patient_id"].isin(prim_ids)
    extra = [c for c in sens.columns if c.startswith("_")]
    sens = sens[MASTER_COLS+extra]
    prim = prim[MASTER_COLS+[c for c in prim.columns if c.startswith("_")]]
    sens.to_csv("funnel1_rebuilt_SENSITIVITY.csv", index=False)
    prim.to_csv("funnel1_rebuilt_PRIMARY.csv", index=False)

    print("\n"+"="*60+"\nFUNNEL 1 REBUILD RESULT\n"+"="*60)
    print(f"  PRIMARY (43644/43775):  {len(prim):,}")
    print(f"  SENSITIVITY (all five): {len(sens):,}")
    print(f"  multi-surgery flagged:  {sens['_multi_surgery_flag'].sum()}")
    print(f"  same-day ambiguous:     {sens['_same_day_ambiguous'].sum()}")
    print(f"  unlisted flagged:       {sens['_unlisted_flag'].sum()}")
    print("  (existing file=1,118; primary ~1,101, sensitivity ~1,118 expected)")
    print("  Demographic/medication columns BLANK -> fill via downstream join.")
    print("\n  >>> READ THE 'PARSING AUDIT' SECTIONS ABOVE before trusting counts. <<<")
    print("  If any audit line shows '+N extra patients', a real format variant")
    print("  exists and we should decide whether to accept it.")

if __name__=="__main__":
    main()
