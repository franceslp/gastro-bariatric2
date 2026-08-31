#!/usr/bin/env python3
"""
step1_demographics_prokinetic_join.py  (FINAL: 4 passes, verified columns)

funnel1_rebuilt_SENSITIVITY.csv -> master_cohort_rebuilt_FINAL.csv

PASS A  patient.csv              demographics (sex, race, ethnicity, marital, yob, death)
PASS B  medication_ingredient    4 prokinetics (first after FIRST-EVER K31.84) + erythromycin routes
PASS C  procedure.csv            GES 78264/78265/78266: days_GES_before_K3184 + days_GES_before_surgery
PASS D  diagnosis.csv            full E10/E11 history -> diabetes_type_label (Type 1 / Type 2 / Both)
                                  (closest_E10_E11_code_before_surgery kept from rebuild, separate col)

Fixes applied: prints columns per pass; QA value_counts at end; deceased from death presence;
age approximated from year_of_birth (TriNetX has no month/day -- labeled 'approx').

Usage:  nohup python3 step1_demographics_prokinetic_join.py > step1_join_log.txt 2>&1 &
"""

import subprocess
import pandas as pd
from collections import defaultdict

BASE = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PATIENT_URI   = f"{BASE}/patient.csv"
MED_ING_URI   = f"{BASE}/medication_ingredient.csv"
PROCEDURE_URI = f"{BASE}/procedure.csv"
DIAGNOSIS_URI = f"{BASE}/diagnosis.csv"

COHORT_CSV = "funnel1_rebuilt_SENSITIVITY.csv"
OUTPUT_CSV = "master_cohort_rebuilt_FINAL.csv"

PROKINETICS = {"6915":"metoclopramide","4053":"erythromycin","3626":"domperidone","2107310":"prucalopride"}
ERYTHRO = "4053"
GES_CODES = {"78264","78265","78266"}


def stream(uri, chunksize=500_000):
    proc = subprocess.Popen(["gsutil","cat",uri], stdout=subprocess.PIPE)
    for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=chunksize):
        yield chunk
    proc.stdout.close(); proc.wait()


def parse_date(s):
    d = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if hasattr(d,"isna") and d.isna().all():
        d = pd.to_datetime(s, errors="coerce")
    return d


def norm_cpt(v):
    s=str(v).strip().upper()
    return s[:-2] if s.endswith(".0") else s


def norm_icd(v):
    import re
    return re.sub(r"\.","",str(v).strip().upper())


def main():
    cohort = pd.read_csv(COHORT_CSV, dtype=str)
    ids = set(cohort["patient_id"])
    print(f"Cohort patients to enrich: {len(ids):,}", flush=True)

    first_k, closest_k, surg_date = {}, {}, {}
    for _, r in cohort.iterrows():
        pid=r["patient_id"]
        ds=pd.to_datetime([d.strip() for d in str(r.get("all_K31_84_dates","")).split(",") if d.strip()], errors="coerce")
        ds=ds[~pd.isna(ds)]
        if len(ds): first_k[pid]=ds.min()
        ck=pd.to_datetime(r.get("closest_K31_84_strictly_before_surgery"), errors="coerce")
        if pd.notna(ck): closest_k[pid]=ck
        sd=pd.to_datetime(r.get("bariatric_date"), errors="coerce")
        if pd.notna(sd): surg_date[pid]=sd

    # ---- PASS A ----
    print("\nPASS A: patient.csv", flush=True)
    demo={}; printed=False
    for chunk in stream(PATIENT_URI):
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed=True
        sub=chunk[chunk["patient_id"].isin(ids)]
        for _, r in sub.iterrows():
            demo[r["patient_id"]]={"sex":r.get("sex",""),"race":r.get("race",""),
                "ethnicity":r.get("ethnicity",""),"marital_status":r.get("marital_status",""),
                "year_of_birth":r.get("year_of_birth",""),
                "month_year_death":r.get("month_year_death","") if pd.notna(r.get("month_year_death")) else ""}
    print(f"  demographics for {len(demo):,}/{len(ids):,}", flush=True)

    # ---- PASS B ----
    print("\nPASS B: medication_ingredient.csv", flush=True)
    first_prok={}; erythro_routes=defaultdict(set); rows=0; printed=False
    for chunk in stream(MED_ING_URI):
        rows+=len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed=True
        sub=chunk[chunk["patient_id"].isin(ids)]
        sub=sub[(sub["code_system"]=="RxNorm")&(sub["code"].isin(PROKINETICS))]
        if sub.empty:
            if rows%50_000_000==0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub=sub.copy(); sub["_d"]=parse_date(sub["start_date"])
        for _, r in sub.iterrows():
            pid=r["patient_id"]; dt=r["_d"]; fk=first_k.get(pid)
            if pd.isna(dt) or fk is None or dt<fk: continue
            drug=PROKINETICS[r["code"]]
            if pid not in first_prok or dt<first_prok[pid][0]:
                first_prok[pid]=(dt,drug)
            if r["code"]==ERYTHRO:
                rt=str(r.get("route","")).strip()
                erythro_routes[pid].add(rt if rt else "unknown")
        if rows%50_000_000==0: print(f"  ...{rows:,} rows, {len(first_prok):,} prok pts", flush=True)
    print(f"  prokinetic-after-dx for {len(first_prok):,} patients", flush=True)

    # ---- PASS C ----
    print("\nPASS C: procedure.csv (GES)", flush=True)
    ges=defaultdict(list); rows=0; printed=False
    for chunk in stream(PROCEDURE_URI):
        rows+=len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed=True
        sub=chunk[chunk["patient_id"].isin(ids)].copy()
        if sub.empty:
            if rows%50_000_000==0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub["_c"]=sub["code"].map(norm_cpt)
        g=sub[sub["_c"].isin(GES_CODES)]
        if not g.empty:
            d=parse_date(g["date"])
            for pid,dt,code in zip(g["patient_id"],d,g["_c"]):
                if pd.notna(dt): ges[pid].append((dt,code))
        if rows%50_000_000==0: print(f"  ...{rows:,} rows, {len(ges):,} GES pts", flush=True)
    print(f"  GES for {len(ges):,} patients", flush=True)

    # ---- PASS D: full E10/E11 history before surgery ----
    print("\nPASS D: diagnosis.csv (E10/E11 history)", flush=True)
    has_e10=set(); has_e11=set(); rows=0; printed=False
    for chunk in stream(DIAGNOSIS_URI):
        rows+=len(chunk)
        if not printed:
            print(f"  columns: {chunk.columns.tolist()}", flush=True); printed=True
        sub=chunk[chunk["patient_id"].isin(ids)].copy()
        if sub.empty:
            if rows%50_000_000==0: print(f"  ...{rows:,} rows", flush=True)
            continue
        sub["_i"]=sub["code"].map(norm_icd)
        sub["_d"]=parse_date(sub["date"])
        # before surgery
        sub["_surg"]=sub["patient_id"].map(surg_date)
        sub=sub[sub["_d"]<sub["_surg"]]
        for pid in sub[sub["_i"].str.startswith("E10",na=False)]["patient_id"]: has_e10.add(pid)
        for pid in sub[sub["_i"].str.startswith("E11",na=False)]["patient_id"]: has_e11.add(pid)
        if rows%50_000_000==0: print(f"  ...{rows:,} rows | E10={len(has_e10):,} E11={len(has_e11):,}", flush=True)
    print(f"  E10-ever-before-surgery: {len(has_e10):,}; E11: {len(has_e11):,}", flush=True)

    # ---- ASSEMBLE ----
    print("\nAssembling...", flush=True)
    out=cohort.copy()
    out["days_GES_before_surgery"]=""
    out["closest_GES_before_surgery_code"]=""
    out["erythromycin_routes_after_dx"]=""
    for i, r in out.iterrows():
        pid=r["patient_id"]; d=demo.get(pid,{}); yob=d.get("year_of_birth","")
        out.at[i,"year_of_birth"]=yob
        out.at[i,"sex"]=d.get("sex","")
        out.at[i,"race"]=d.get("race","")
        out.at[i,"ethnicity"]=d.get("ethnicity","")
        out.at[i,"marital_status"]=d.get("marital_status","")
        myd=d.get("month_year_death","")
        out.at[i,"month_year_death"]=myd
        out.at[i,"deceased"]="True" if myd else "False"
        try:
            bdate=pd.to_datetime(r["bariatric_date"]); age=bdate.year-int(yob)
            out.at[i,"age_at_surgery_approx"]=age
            out.at[i,"meets_age_requirement"]="True" if age>=18 else "False"
        except Exception:
            out.at[i,"age_at_surgery_approx"]=""; out.at[i,"meets_age_requirement"]=""
        # diabetes_type_label from FULL history
        e10=pid in has_e10; e11=pid in has_e11
        out.at[i,"diabetes_type_label"]="Both" if (e10 and e11) else ("Type 1" if e10 else ("Type 2" if e11 else ""))
        # prokinetic
        if pid in first_prok:
            dt,drug=first_prok[pid]; fk=first_k.get(pid)
            out.at[i,"first_prokinetic_after_K3184_dx_drug"]=drug
            out.at[i,"first_prokinetic_after_K3184_dx_date"]=dt.date()
            out.at[i,"days_to_prokinetic_after_K3184"]=(dt-fk).days if fk is not None else ""
        else:
            out.at[i,"first_prokinetic_after_K3184_dx_drug"]=""
            out.at[i,"first_prokinetic_after_K3184_dx_date"]=""
            out.at[i,"days_to_prokinetic_after_K3184"]=""
        out.at[i,"erythromycin_routes_after_dx"]="|".join(sorted(erythro_routes.get(pid,set())))
        # GES both timings
        gd=ges.get(pid,[]); ck=closest_k.get(pid); sd=surg_date.get(pid)
        if gd and ck is not None:
            bk=[(x,c) for (x,c) in gd if x<=ck]
            if bk:
                g_dt,g_code=max(bk,key=lambda t:t[0])
                out.at[i,"closest_GES_before_K3184_dx_date"]=g_dt.date()
                out.at[i,"closest_GES_before_K3184_dx_code"]=g_code
                out.at[i,"days_GES_before_K3184"]=(ck-g_dt).days
        if gd and sd is not None:
            bs=[(x,c) for (x,c) in gd if x<sd]
            if bs:
                g_dt,g_code=max(bs,key=lambda t:t[0])
                out.at[i,"days_GES_before_surgery"]=(sd-g_dt).days
                out.at[i,"closest_GES_before_surgery_code"]=g_code

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out):,} rows)", flush=True)

    # ---- QA ----
    print("\n--- QA / fill rates ---", flush=True)
    for c in ["sex","race","year_of_birth","diabetes_type_label",
              "first_prokinetic_after_K3184_dx_drug","erythromycin_routes_after_dx",
              "days_GES_before_K3184","days_GES_before_surgery","deceased"]:
        if c in out.columns:
            filled=(out[c].astype(str).str.len()>0).sum()
            print(f"  {c}: {filled:,}/{len(out):,} filled", flush=True)
    print("\ndiabetes_type_label distribution (expect ~Type2=941, Both=168, Type1=9):", flush=True)
    print(out["diabetes_type_label"].value_counts(dropna=False).to_string(), flush=True)
    print("\nfirst_prokinetic drug distribution:", flush=True)
    print(out["first_prokinetic_after_K3184_dx_drug"].value_counts(dropna=False).to_string(), flush=True)
    print("\nclosest GES before K31.84 code distribution:", flush=True)
    print(out["closest_GES_before_K3184_dx_code"].value_counts(dropna=False).to_string(), flush=True)
    print("\nerythromycin routes distribution:", flush=True)
    print(out["erythromycin_routes_after_dx"].value_counts(dropna=False).to_string(), flush=True)


if __name__=="__main__":
    main()
