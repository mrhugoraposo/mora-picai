# Deploying the MoRA Research Console — step by step

This is the complete install / configure / run / deploy guide for the **MoRA Research
Console** (`services/api/`), the authenticated FastAPI web app that runs the Modality-Aware
Reliability (MoRA) workflow live. Three paths are documented:

| Path | For | Data needed | Time |
|---|---|---|---|
| **A — Synthetic demo** | trying it / reviewers | none | ~3 min |
| **B — Local, real models** | running on your own PI-CAI data | PI-CAI | ~varies |
| **C — Docker + Postgres** | self-hosted deployment | real models + data | ~10 min |

> **Security first.** The console enforces login but is **not hardened for the open
> internet**. Bind it to localhost/LAN and (for any networked use) put it behind HTTPS via a
> reverse proxy or a VPN. **Never** port-forward it to the public internet, and never load
> patient data onto a publicly reachable instance. See [§6](#6-security-hardening).

---

## 1. Prerequisites

- **Python 3.12** (for Paths A/B). Check: `python3.12 --version`.
- **git**.
- **Docker + Docker Compose** (Path C only).
- A modern web browser.
- The app does **CPU-only** inference — no GPU/PyTorch is needed at runtime.

Clone the repository:
```bash
git clone https://github.com/mrhugoraposo/mora-picai.git
cd mora-picai
```

---

## 2. How it fits together (read once)

- `services/api/` is the FastAPI app (`app/main.py` = routes + auth; `app/mora_engine.py` =
  the live MoRA mechanism; `app/db.py` = SQLAlchemy; Jinja2 templates).
- Inference loads model artifacts from `services/ai/califusion-cnn/models/` (built from real
  data by `scripts/build_deployment_models.py`). **If those are absent, the engine
  automatically falls back to the bundled SYNTHETIC demo assets** in
  `services/api/demo_assets/` (built by `services/api/build_demo.py`).
- **Database:** SQLite by default (`services/api/mora_console.db`, zero-config); **Postgres**
  when `DATABASE_URL` is set (Path C).
- **Auth:** username/password, bcrypt-hashed; itsdangerous-signed httponly session cookies
  (8 h). The admin user is seeded from env vars on every boot. Every route requires login
  except `/login`, `/health`, `/static/*`.

---

## 3. Path A — Synthetic demo (no data, no Docker)

The fastest way to see the full workflow on **non-identifiable, randomly-generated** data.

```bash
# 3.1 — create the virtual environment where the app expects it, and install deps
python3.12 -m venv services/ai/califusion-cnn/.venv
source services/ai/califusion-cnn/.venv/bin/activate
pip install --upgrade pip
pip install -r services/api/requirements.txt

# 3.2 — build the synthetic demo models (writes services/api/demo_assets/)
cd services/api
python build_demo.py

# 3.3 — run it (SQLite, port 8080)
./run_local.sh 8080
```

Open **http://127.0.0.1:8080** and sign in with the default credentials **`admin` / `changeme`**.
Go to **Inference → REPLAY**, pick a `DEMO-###` case, and toggle **break imaging** — watch
`r_img` collapse and MoRA down-weight imaging (or defer). All demo data is synthetic; no
patient data is involved.

---

## 4. Path B — Local run with real models (SQLite)

Use this if you have the PI-CAI data and have run the pipeline.

```bash
# 4.1 — environment (same as 3.1)
python3.12 -m venv services/ai/califusion-cnn/.venv
source services/ai/califusion-cnn/.venv/bin/activate
pip install --upgrade pip
pip install -r services/api/requirements.txt

# 4.2 — obtain PI-CAI and build the pipeline's processed features
#        (see services/ai/califusion-cnn/data/README_DATA.md for acquisition + layout)

# 4.3 — build the deployment models the app loads
cd services/ai/califusion-cnn
python scripts/build_deployment_models.py
#  -> writes services/ai/califusion-cnn/models/{*.joblib, *_replay.csv, manifest.json}

# 4.4 — run it
cd ../../api
./run_local.sh 8080
```

When real `models/` exist, the engine uses them; otherwise it falls back to the synthetic
demo from Path A.

---

## 5. Path C — Docker Compose + Postgres (self-hosted)

For a persistent, Postgres-backed deployment with **real** models. (For a data-free trial,
use Path A — Compose mounts real models/data read-only and is not meant for the synthetic demo.)

```bash
# 5.1 — configure secrets
cd infrastructure/docker
cp .env.example .env
#  edit .env and set STRONG values for:
#     SECRET_KEY            (e.g.  python -c "import secrets; print(secrets.token_urlsafe(48))")
#     APP_ADMIN_PASSWORD
#     POSTGRES_PASSWORD

# 5.2 — ensure the real model + data artifacts exist on the host (mounted read-only):
#     services/ai/califusion-cnn/models/                     (built in Path B, step 4.3)
#     services/ai/califusion-cnn/data/processed/
#     services/ai/califusion-cnn/data/raw/PI-CAI/marksheet.csv

# 5.3 — build + start (api + postgres)
docker compose up -d
docker compose logs -f api      # watch boot; admin user is seeded on startup
```

The app listens on **`127.0.0.1:8000`** (loopback only). Open **http://127.0.0.1:8000** and
sign in. Postgres is **not** published to the host — only the api service reaches it on the
compose network. The DB volume (`mora_pgdata`) persists across restarts.

Stop / reset:
```bash
docker compose down        # stop
docker compose down -v     # stop AND delete the Postgres volume (wipes users + run history)
```

---

## 6. Security hardening

The console is login-protected but **not hardened for the open internet**. For any use beyond
localhost:

- **Bind to LAN/loopback only.** Compose already binds `127.0.0.1:8000`; for `run_local.sh`
  it is `127.0.0.1` too. Do **not** port-forward the port to the internet.
- **Terminate HTTPS with a reverse proxy** (Caddy for automatic TLS, or nginx) **or use a
  VPN** (WireGuard/Tailscale). Session cookies are httponly but are only safe over HTTPS.
  Example `Caddyfile`:
  ```
  mora.example.lan {
      reverse_proxy 127.0.0.1:8000
  }
  ```
- **Set strong secrets** (`SECRET_KEY`, `APP_ADMIN_PASSWORD`, `POSTGRES_PASSWORD`). The app
  prints a warning at startup if the default password `changeme` is in use.
- **Never commit** `.env`, the DB, `models/`, or uploads (all git-ignored).
- **Do not load patient data onto a publicly reachable instance** (PI-CAI is CC BY-NC).

---

## 7. Configuration reference (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | `dev-insecure-change-me` | Signs session cookies. **Set a strong value in production.** |
| `APP_ADMIN_USER` | `admin` | Seeded admin username (re-applied on every boot). |
| `APP_ADMIN_PASSWORD` | `changeme` | Seeded admin password. **Set a strong value in production.** |
| `DATABASE_URL` | _(unset → SQLite)_ | Set to a `postgresql+psycopg2://…` URL for Postgres. |
| `MODELS_DIR` | `…/califusion-cnn/models` | Where the app loads model artifacts; auto-falls back to `services/api/demo_assets/`. |
| `CALIFUSION_PIPELINE_ROOT` | `…/califusion-cnn` | Pipeline root (used to resolve `models/`). |
| `PORT` | `8080` (`run_local.sh`) | Local dev port. Compose uses `8000`. |

---

## 8. Verify it's working

```bash
curl -s http://127.0.0.1:8080/health      # -> {"status":"ok","models_built":true|false}
```
Then in the browser: **Dashboard** (datasets + recent runs) → **Inference** (replay a case,
toggle break-imaging) → **Experiments** (run a live experiment) → **History** (audit trail).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Login page, but inference returns **503 / "models not built"**; startup logs `models loaded for: NONE` | Build models: `python services/api/build_demo.py` (synthetic) or `scripts/build_deployment_models.py` (real). |
| `run_local.sh`: `No such file …/.venv/bin/python` | Create the venv at `services/ai/califusion-cnn/.venv` (step 3.1). |
| Port already in use | `./run_local.sh <other-port>`, or change the Compose port mapping. |
| Too many failed logins / 429 | Login is rate-limited (5 fails / 5 min per user+IP); wait, and confirm `APP_ADMIN_PASSWORD`. |
| Compose `db` unhealthy | Ensure `POSTGRES_PASSWORD` is set in `.env`. |
| Compose fails to mount `marksheet.csv` / `models` | Those paths must exist on the host (Path B first); Compose is for real-data deployment, not the synthetic demo. |
