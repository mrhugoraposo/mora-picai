#!/usr/bin/env python3
"""Phase A feasibility audit — NSCLC-Radiogenomics external-set viability (Gate 2)."""
import json, ssl, urllib.parse, urllib.request, collections, sys
BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
CTX = ssl.create_default_context(); CTX.check_hostname=False; CTX.verify_mode=ssl.CERT_NONE
def get(ep, params=None):
    url = BASE+ep+("?"+urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent":"califusion-feasibility/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=120, context=CTX).read().decode())

# 1) find the exact collection name
cols = get("getCollectionValues")
names = [c.get("Collection") for c in cols]
matches = [n for n in names if n and "radiogenom" in n.lower()]
print("Collections matching 'radiogenom':", matches)
lung = [n for n in names if n and ("nsclc" in n.lower() or "lung" in n.lower())]
print("Lung/NSCLC collections:", lung)

print("\n=== NSCLC Radiogenomics imaging enumeration ===")
COL = "NSCLC Radiogenomics"
series = get("getSeries", {"Collection": COL})
by_mod = collections.Counter(s.get("Modality") for s in series)
print("series by modality:", dict(by_mod))
pts_all = set(s.get("PatientID") for s in series)
print("unique patients (any imaging):", len(pts_all))
# segmentation coverage
seg_mods = {"SEG","RTSTRUCT"}
pts_seg = set(s.get("PatientID") for s in series if s.get("Modality") in seg_mods)
pts_ct  = set(s.get("PatientID") for s in series if s.get("Modality")=="CT")
pts_pet = set(s.get("PatientID") for s in series if s.get("Modality") in ("PT","PET"))
print("patients with CT:", len(pts_ct))
print("patients with PET/PT:", len(pts_pet))
print("patients with SEG/RTSTRUCT (tumor mask):", len(pts_seg))
print("patients with CT AND mask:", len(pts_ct & pts_seg))
# persist patient lists for intersection with clinical
import os, json as J
os.makedirs("results/feasibility", exist_ok=True)
J.dump({"collection":COL,"modalities":dict(by_mod),
        "patients_any":sorted(pts_all),"patients_ct":sorted(pts_ct),
        "patients_seg":sorted(pts_seg),"patients_ct_and_seg":sorted(pts_ct & pts_seg)},
       open("results/feasibility/radiogenomics_imaging.json","w"), indent=1)
print("\nsample patient IDs:", sorted(pts_all)[:5])

print("\n=== Radiogenomics clinical file ===")
import io, pandas as pd, numpy as np
CSV="https://www.cancerimagingarchive.net/wp-content/uploads/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
req=urllib.request.Request(CSV, headers={"User-Agent":"califusion/1.0"})
raw=urllib.request.urlopen(req,timeout=90,context=CTX).read().decode("utf-8-sig","replace")
df=pd.read_csv(io.StringIO(raw))
print("clinical rows:", len(df), "| columns:", len(df.columns))
print("\nColumn names:")
for c in df.columns: print("  -", repr(c))
print("\nFirst Case IDs:", df.iloc[:,0].astype(str).head(6).tolist())
df.to_csv("results/feasibility/radiogenomics_clinical_raw.csv", index=False)

print("\n=== Survival fields inspection ===")
ss = df['Survival Status'].astype(str).str.strip()
print("Survival Status:", dict(ss.value_counts(dropna=False)))
ttd = pd.to_numeric(df['Time to Death (days)'], errors='coerce')
print("Time to Death (days): non-null =", ttd.notna().sum(),
      "| min/median/max =", (np.nanmin(ttd), np.nanmedian(ttd), np.nanmax(ttd)))
print("Sample dates - CT Date:", df['CT Date'].astype(str).head(3).tolist(),
      "| Last Known Alive:", df['Date of Last Known Alive'].astype(str).head(3).tolist())

# Build follow-up time: dead -> Time to Death; alive -> (Last Known Alive - CT Date)
def parse(d): return pd.to_datetime(d, errors='coerce')
ct = parse(df['CT Date']); lka = parse(df['Date of Last Known Alive']); dod = parse(df['Date of Death'])
is_dead = ss.str.lower().eq('dead')
fu_alive_days = (lka - ct).dt.days
# survival time: prefer Time to Death for dead; else days from CT to last-known-alive
surv_time = ttd.where(is_dead, fu_alive_days)
print("\nDerived survival time: non-null =", surv_time.notna().sum(),
      "| alive follow-up median days =", np.nanmedian(fu_alive_days[~is_dead]))

H=730
pos = is_dead & (surv_time <= H)
neg = surv_time > H
censored_lt2y = (~is_dead) & (surv_time <= H)
usable = pos | neg
print("\n=== 2-year OS labeling (clinical, all 211) ===")
print("dead total:", int(is_dead.sum()))
print("positive (dead <=730d):", int(pos.sum()))
print("negative (followed >730d):", int(neg.sum()))
print("excluded (alive but censored <2y):", int(censored_lt2y.sum()))
print("unusable (missing dates/time):", int((~usable & ~censored_lt2y).sum()))
print("USABLE total:", int(usable.sum()), "| prevalence(pos):",
      round(float(pos.sum())/max(1,int(usable.sum())),3))

# Intersect with imaging
import json as J
img = J.load(open("results/feasibility/radiogenomics_imaging.json"))
ids_ct = set(img["patients_ct"]); ids_seg = set(img["patients_seg"])
case = df['Case ID'].astype(str).str.strip()
df2 = df.assign(_pos=pos.values, _neg=neg.values, _usable=usable.values, _case=case.values)
use_df = df2[df2['_usable']]
in_ct  = use_df[use_df['_case'].isin(ids_ct)]
in_seg = use_df[use_df['_case'].isin(ids_seg)]
print("\n=== Usable INTERSECTED with imaging ===")
print("usable & has CT:", len(in_ct), "| positives:", int(in_ct['_pos'].sum()))
print("usable & has SEG mask:", len(in_seg), "| positives:", int(in_seg['_pos'].sum()),
      "| negatives:", int(in_seg['_neg'].sum()))
print("   -> prevalence on SEG-covered usable:",
      round(int(in_seg['_pos'].sum())/max(1,len(in_seg)),3))

summary={
 "collection":"NSCLC Radiogenomics","horizon_days":H,
 "clinical_rows":len(df),"dead_total":int(is_dead.sum()),
 "usable_clinical":int(usable.sum()),"positives_clinical":int(pos.sum()),
 "ct_patients":len(ids_ct),"seg_patients":len(ids_seg),
 "usable_with_ct":len(in_ct),"positives_with_ct":int(in_ct['_pos'].sum()),
 "usable_with_seg":len(in_seg),"positives_with_seg":int(in_seg['_pos'].sum()),
 "negatives_with_seg":int(in_seg['_neg'].sum()),
 "prevalence_seg":round(int(in_seg['_pos'].sum())/max(1,len(in_seg)),3),
 "license":"CC BY 3.0"
}
J.dump(summary, open("results/feasibility/GATE2_summary.json","w"), indent=2)
print("\n=== GATE 2 SUMMARY ==="); print(J.dumps(summary, indent=2))
