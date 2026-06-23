"""
main.py — FastAPI app for the MoRA Research Console (auth + live inference + experiments).

Serves the frontend via Jinja2 (Tailwind + Chart.js CDN). One robust service + DB:
  - SQLite by default, Postgres when DATABASE_URL is set
  - session-cookie auth; EVERY route requires login except /login, /health, /static
  - POST /infer        : live MoRA on a replay case or an upload (centrepiece)
  - POST /experiments/run : run an experiment live on cached features
  - run history persisted to the DB (audit trail)
"""
from __future__ import annotations
import os
import sys
import traceback

import numpy as np
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from . import auth, experiments as EXP
from .db import init_db, get_session, Run, User, record_run
from . import mora_engine as ENGINE
from . import upload as UPLOAD

app = FastAPI(title="MoRA Research Console")
templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")

PUBLIC_PATHS = {"/login", "/health", "/favicon.ico"}


@app.on_event("startup")
def _startup():
    init_db()
    user = auth.seed_admin()
    msg = f"[startup] DB ready; admin user '{user}'"
    if auth.used_default_password():
        msg += "  ** USING DEFAULT PASSWORD 'changeme' — set APP_ADMIN_PASSWORD in production **"
    print(msg)
    try:
        ds = ENGINE.datasets_available()
        print(f"[startup] models loaded for: {list(ds.keys()) or 'NONE — run build_deployment_models.py'}")
    except Exception as e:
        print(f"[startup] model load warning: {e}")


# ----------------------------------------------------------------- auth middleware
@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)
    sess = auth.read_session_cookie(request.cookies.get(auth.COOKIE_NAME, ""))
    if not sess:
        if path.startswith("/api") or request.headers.get("accept", "").startswith("application/json"):
            return JSONResponse({"error": "auth required"}, status_code=401)
        return RedirectResponse("/login", status_code=302)
    request.state.user = sess
    return await call_next(request)


def current_user(request: Request) -> dict:
    return getattr(request.state, "user", None)


def _jsonsafe(o):
    """Recursively coerce numpy scalars/arrays to plain Python for JSON responses."""
    if isinstance(o, dict):
        return {k: _jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonsafe(v) for v in o]
    if isinstance(o, np.generic):
        v = o.item()
        return None if (isinstance(v, float) and np.isnan(v)) else v
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ----------------------------------------------------------------- public routes
@app.get("/health")
def health():
    ok_models = bool(ENGINE.datasets_available())
    return {"status": "ok", "models_built": ok_models}


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: str = ""):
    if auth.read_session_cookie(request.cookies.get(auth.COOKIE_NAME, "")):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    ip = request.client.host if request.client else "?"
    key = f"{username}|{ip}"
    if auth.rate_limited(key):
        return templates.TemplateResponse(request, "login.html", {
            "error": "Too many failed attempts. Wait a few minutes and try again."},
            status_code=429)
    user = auth.authenticate(username, password)
    if not user:
        auth.record_fail(key)
        return templates.TemplateResponse(request, "login.html",
            {"error": "Invalid username or password."}, status_code=401)
    auth.clear_fails(key)
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie(auth.COOKIE_NAME, auth.make_session_cookie(user), httponly=True,
                    samesite="lax", max_age=auth.SESSION_MAX_AGE)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


# ----------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_session)):
    ds = ENGINE.datasets_available()
    recent = db.query(Run).order_by(Run.created_at.desc()).limit(8).all()
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": current_user(request),
        "datasets": ds, "recent": recent,
        "manifest": ENGINE.manifest() if ds else None})


@app.get("/infer", response_class=HTMLResponse)
def infer_page(request: Request):
    ds = ENGINE.datasets_available()
    cases = {d: ENGINE.list_cases(d, 30) for d in ds} if ds else {}
    return templates.TemplateResponse(request, "infer.html", {
        "user": current_user(request),
        "datasets": ds, "cases": cases,
        "clinical_schemas": UPLOAD.CLINICAL_SCHEMAS})


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request, db: Session = Depends(get_session)):
    recent = db.query(Run).filter(Run.kind == "experiment").order_by(
        Run.created_at.desc()).limit(6).all()
    return templates.TemplateResponse(request, "experiments.html", {
        "user": current_user(request),
        "experiments": EXP.EXPERIMENTS, "recent": recent})


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, db: Session = Depends(get_session)):
    runs = db.query(Run).order_by(Run.created_at.desc()).limit(100).all()
    users = {u.id: u.username for u in db.query(User).all()}
    return templates.TemplateResponse(request, "history.html", {
        "user": current_user(request),
        "runs": runs, "users": users})


# ----------------------------------------------------------------- inference API
@app.post("/infer")
async def do_infer(
    request: Request,
    mode: str = Form("replay"),
    dataset: str = Form("picai"),
    case_id: str = Form(""),
    break_imaging: str = Form(""),
    severity: float = Form(4.0),
    radiomics_vector: str = Form(""),
    imaging_file: UploadFile = File(None),
    db: Session = Depends(get_session),
):
    u = current_user(request)
    try:
        if mode == "replay":
            res = ENGINE.infer_replay(dataset, case_id,
                                      break_imaging=bool(break_imaging), severity=severity)
        else:  # upload
            res = await UPLOAD.handle_upload(dataset, request, imaging_file,
                                             radiomics_vector)
        res = _jsonsafe(res)
        record_run(db, u["uid"] if u else None, "inference", dataset,
                   str(res.get("case_id", res.get("source", "upload"))),
                   {"p_mora": res["p_mora"], "p_img": res["p_img"], "p_clin": res["p_clin"],
                    "r_img": res["r_img"], "r_clin": res["r_clin"],
                    "decision": res["decision"], "label": res.get("case_label")})
        return JSONResponse(res)
    except ENGINE.ModelsNotBuilt as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)


# ----------------------------------------------------------------- experiments API
@app.post("/experiments/run")
def do_experiment(request: Request, name: str = Form(...),
                  seeds: int = Form(10), severity: float = Form(4.0),
                  db: Session = Depends(get_session)):
    u = current_user(request)
    try:
        kw = {"seeds": int(seeds), "severity": float(severity)} if name == "mora_failure" else {}
        res = _jsonsafe(EXP.run_experiment(name, **kw))
        record_run(db, u["uid"] if u else None, "experiment",
                   EXP.EXPERIMENTS[name]["dataset"], name,
                   {"headline": res["headline"], "verdict": res["verdict"],
                    "runtime_s": res["runtime_s"]})
        return JSONResponse(res)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)


@app.get("/case_features")
def case_features(dataset: str, case_id: str):
    """Return a case's raw clinical features to pre-fill the upload form (replay convenience)."""
    try:
        df = ENGINE.replay_table(dataset)
        idcol = "patient_id" if dataset == "picai" else "PatientID"
        row = df[df[idcol] == str(case_id)]
        if row.empty:
            return JSONResponse({"error": "not found"}, status_code=404)
        r = row.iloc[0]
        if dataset == "picai":
            cols = ["patient_age", "psa", "psad", "prostate_volume"]
        else:
            cols = ["age", "gender", "overall_stage", "histology"]

        def _coerce(v):
            if v is None:
                return None
            if isinstance(v, (np.floating, float)):
                return None if np.isnan(v) else float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            return str(v)
        return {c: _coerce(r.get(c)) for c in cols}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
