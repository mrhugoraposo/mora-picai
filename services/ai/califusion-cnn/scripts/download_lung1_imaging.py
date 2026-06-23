#!/usr/bin/env python3
"""
scripts/download_lung1_imaging.py  —  bulk CT + RTSTRUCT download for Lung1 (Phase B).

Pulls the canonical primary-tumour pipeline inputs only:
  * CT       (~27 GB)  — pre-treatment axial CT series, one per patient
  * RTSTRUCT (~0.6 GB) — GTV-1 contour (canonical Aerts segmentation; parsed by rt-utils)
SEG (~8 GB) is intentionally skipped — RTSTRUCT GTV-1 is the mask source (validated in
Phase B smoke test); SEG remains available as a documented fallback.

Layout (matches tcia_masks.find_series_dirs):
  data/raw/NSCLC-Radiomics/<PatientID>/<MOD>_<uid_tail>/<*.dcm>

Resumable: a series whose destination dir already has >=1 .dcm is skipped. Per-series
retry with backoff; failures are recorded, not fatal. Writes download_manifest.json.

Run:  python scripts/download_lung1_imaging.py            # full pull
      python scripts/download_lung1_imaging.py --limit 5  # first 5 patients (smoke)
      python scripts/download_lung1_imaging.py --modalities CT  # CT only
"""
from __future__ import annotations
import argparse
import glob
import io
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
COLLECTION = "NSCLC-Radiomics"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "raw", COLLECTION)


def _get(endpoint, params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "califusion/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=120, context=CTX).read().decode())


def dest_for(s):
    pid = s.get("PatientID", "UNK"); mod = s.get("Modality"); uid = s["SeriesInstanceUID"]
    return os.path.join(OUT, pid, f"{mod}_{uid[-12:]}")


def already_have(dest):
    return os.path.isdir(dest) and any(f.endswith(".dcm") for f in os.listdir(dest))


def download_series(uid, dest, retries=3):
    last = None
    for attempt in range(retries):
        try:
            os.makedirs(dest, exist_ok=True)
            url = BASE + "getImage?" + urllib.parse.urlencode({"SeriesInstanceUID": uid})
            req = urllib.request.Request(url, headers={"User-Agent": "califusion/1.0"})
            data = urllib.request.urlopen(req, timeout=900, context=CTX).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest)
            return len(data)
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", nargs="+", default=["CT", "RTSTRUCT"])
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (0=all)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    series = _get("getSeries", {"Collection": COLLECTION})
    wanted = [s for s in series if s.get("Modality") in set(args.modalities)]
    if args.limit:
        keep_pids = sorted({s.get("PatientID") for s in wanted})[:args.limit]
        wanted = [s for s in wanted if s.get("PatientID") in set(keep_pids)]
    # CT first (largest), then RTSTRUCT; group so a patient completes together-ish
    wanted.sort(key=lambda s: (s.get("PatientID"), s.get("Modality") != "CT"))

    total = len(wanted)
    est_gb = sum(float(s.get("FileSize", 0)) for s in wanted) / 1e9
    print(f"[{time.strftime('%H:%M:%S')}] target: {total} series "
          f"({', '.join(args.modalities)}) ~{est_gb:.1f} GB -> {OUT}", flush=True)

    manifest = []
    done = skipped = failed = 0
    bytes_dl = 0
    t0 = time.time()
    for i, s in enumerate(wanted, 1):
        pid = s.get("PatientID"); mod = s.get("Modality"); uid = s["SeriesInstanceUID"]
        dest = dest_for(s)
        if already_have(dest):
            skipped += 1
            manifest.append({"pid": pid, "mod": mod, "status": "skip", "dest": dest})
            continue
        try:
            nb = download_series(uid, dest)
            bytes_dl += nb; done += 1
            manifest.append({"pid": pid, "mod": mod, "status": "ok", "mb": round(nb / 1e6, 1)})
        except Exception as e:
            failed += 1
            manifest.append({"pid": pid, "mod": mod, "status": "fail", "error": repr(e)[:140]})
            print(f"  FAIL {pid} {mod}: {repr(e)[:100]}", flush=True)
        if i % 20 == 0 or i == total:
            el = time.time() - t0
            rate = (done + skipped) / el if el else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"  [{time.strftime('%H:%M:%S')}] {i}/{total} "
                  f"ok={done} skip={skipped} fail={failed} "
                  f"{bytes_dl/1e9:.1f}GB {el/60:.1f}min ETA~{eta:.0f}min", flush=True)

    with open(os.path.join(OUT, "download_manifest.json"), "w") as f:
        json.dump({"collection": COLLECTION, "modalities": args.modalities,
                   "n_series": total, "ok": done, "skipped": skipped, "failed": failed,
                   "gb_downloaded": round(bytes_dl / 1e9, 2),
                   "minutes": round((time.time() - t0) / 60, 1),
                   "series": manifest}, f, indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] DONE ok={done} skip={skipped} fail={failed} "
          f"{bytes_dl/1e9:.1f}GB in {(time.time()-t0)/60:.1f}min", flush=True)
    # non-zero exit if anything failed, so the caller can react
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
