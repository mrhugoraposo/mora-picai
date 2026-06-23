#!/usr/bin/env python3
"""
scripts/download_tcia_lung1.py  —  fetch NSCLC-Radiomics (Lung1) imaging.

Uses the public TCIA NBIA REST API to enumerate CT and SEG series for the
NSCLC-Radiomics collection and download them as DICOM. The collection is public
(CC BY-NC 3.0); no API key is required for public series metadata. For bulk
imaging download you may alternatively use the official NBIA Data Retriever or
`nbia-toolkit` (pip install nbiatoolkit) which handles large transfers robustly.

Outputs:
    data/raw/NSCLC-Radiomics/<PatientID>/<SeriesUID>/*.dcm

This script intentionally downloads metadata + provides per-series download URLs;
flip DOWNLOAD=True to pull image archives (large: ~30+ GB total).
"""
from __future__ import annotations
import json, os, ssl, sys, urllib.parse, urllib.request, zipfile, io

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
COLLECTION = "NSCLC-Radiomics"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", COLLECTION)
DOWNLOAD = False                      # set True to actually pull DICOM archives
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE


def _get(endpoint, params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "califusion/1.0"})
    return json.loads(urllib.request.urlopen(req, timeout=90, context=CTX).read().decode())


def list_series():
    series = _get("getSeries", {"Collection": COLLECTION})
    cts = [s for s in series if s.get("Modality") == "CT"]
    segs = [s for s in series if s.get("Modality") in ("SEG", "RTSTRUCT")]
    print(f"{COLLECTION}: {len(series)} series total | CT={len(cts)} | SEG/RTSTRUCT={len(segs)}")
    return cts, segs


def download_series(series_uid, dest):
    os.makedirs(dest, exist_ok=True)
    url = BASE + "getImage?" + urllib.parse.urlencode({"SeriesInstanceUID": series_uid})
    req = urllib.request.Request(url, headers={"User-Agent": "califusion/1.0"})
    data = urllib.request.urlopen(req, timeout=600, context=CTX).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)


def main():
    os.makedirs(OUT, exist_ok=True)
    cts, segs = list_series()
    manifest = {"CT": cts, "SEG": segs}
    with open(os.path.join(OUT, "series_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {os.path.join(OUT, 'series_manifest.json')}")
    if not DOWNLOAD:
        print("DOWNLOAD=False (metadata only). Set DOWNLOAD=True or use nbia-toolkit "
              "to pull DICOM archives (~30+ GB).")
        return
    for s in cts + segs:
        pid = s.get("PatientID", "UNK"); uid = s["SeriesInstanceUID"]
        dest = os.path.join(OUT, pid, uid)
        if os.path.isdir(dest) and os.listdir(dest):
            continue
        try:
            download_series(uid, dest)
            print("downloaded", pid, uid[:16])
        except Exception as e:
            print("FAILED", pid, repr(e)[:100])


if __name__ == "__main__":
    main()
