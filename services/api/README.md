# MoRA Research Console

A self-hostable, **authenticated** web app that runs the **MoRA** (Modality-Reliability
Adaptation) mechanism **live** on submitted or cached medical cases, and runs the headline
experiments live against the cached datasets. It reuses the existing Python pipeline
(`services/ai/califusion-cnn`) — no ML logic is duplicated.

One robust service: **FastAPI** serves both the JSON API and the frontend (Jinja2 + Tailwind
CDN + Chart.js CDN). **SQLite** by default (zero-config local dev), **Postgres** when
`DATABASE_URL` is set (Docker prod).

---

## What it does

- **Live MoRA inference** (`/infer`) — for a case, compute per-modality risk (`p_img`, `p_clin`),
  per-modality **reliability** (`r_img`, `r_clin`; is each modality in-distribution vs the training
  scanners?), the reliability-weighted fused risk
  `p_mora = (r_img·p_img + r_clin·p_clin)/(r_img+r_clin)`, and the **decision**
  (predict / down-weight a modality / **abstain & defer to clinician**) with modality attribution.
  Renders a **step-by-step derivation** of exactly how the output was produced.
  - **REPLAY** mode: pick a cached PI-CAI or Lung1 case by id. Optional toggle corrupts the
    imaging to simulate a scanner shift — watch `r_img` collapse and MoRA down-weight imaging.
  - **UPLOAD** mode: fill the clinical form + (optionally) upload an imaging file (`.mha`/`.nii.gz`)
    for best-effort radiomics extraction, or paste a radiomics feature vector. If extraction is
    infeasible for an arbitrary upload, the app says so and falls back to clinical-only / REPLAY.
- **Live experiments** (`/experiments`) — run on the cached tabular features at request time:
  - **Gate-B non-redundancy** (PI-CAI): clinical vs imaging vs fusion, 5×5 CV AUROC + bootstrap CI.
  - **MoRA vs SOTA under modality failure** (PI-CAI): static / TransCal-CPCS / evidential-TMC / MoRA.
  - **Lung1 redundancy** (negative control): imaging adds no complementary discrimination.
  Each returns a results table, a Chart.js figure, and reproducibility metadata (seeds, n, CIs).
- **Auth + audit** — session-cookie login, bcrypt-hashed passwords, every run (who/what/when/result)
  persisted to the DB and viewable at `/history`.

Every number shown is **computed live** from cached public data with patient-level splits, fixed
seeds and bootstrap intervals. Nothing is fabricated; no patient imaging is served.

---

## Step 0 — build the deployment models (once)

The app loads persisted, source-fit models + MoRA reliability components. Build them in the pipeline:

```bash
cd services/ai/califusion-cnn
.venv/bin/python scripts/build_deployment_models.py
# -> writes services/ai/califusion-cnn/models/{*.joblib,*.csv,manifest.json}
```

This requires the cached feature CSVs already present under
`services/ai/califusion-cnn/data/processed/` (they are, in this repo).

---

## Run locally (no Docker — SQLite)

```bash
cd services/api
# uses the pipeline venv; SQLite db at services/api/mora_console.db
SECRET_KEY=dev-secret APP_ADMIN_USER=admin APP_ADMIN_PASSWORD=changeme \
  ../ai/califusion-cnn/.venv/bin/python -m uvicorn app.main:app --reload --port 8080
# or simply:  ./run_local.sh 8080
```

Open http://127.0.0.1:8080 and sign in with the admin credentials above.

---

## Run on a home server (Docker Compose — Postgres)

```bash
cd infrastructure/docker
cp .env.example .env
#  edit .env:  set SECRET_KEY, APP_ADMIN_PASSWORD, POSTGRES_PASSWORD (strong values)
#  generate a key:  python -c "import secrets; print(secrets.token_urlsafe(48))"

docker compose up -d            # builds the image, starts postgres + the api
docker compose logs -f api      # watch boot; the admin user is seeded on startup
```

The app listens on `127.0.0.1:8000` by default. The prebuilt `models/` and cached
`data/processed/` are mounted **read-only** (never baked into the image, never committed).

### Setting / changing the admin password
The admin user is (re)created from `APP_ADMIN_USER` / `APP_ADMIN_PASSWORD` on **every boot**.
To change it: edit `.env`, then `docker compose up -d` (or `docker compose restart api`).

---

## Auth model

- Username/password login; passwords hashed with **passlib bcrypt**.
- Sessions are **itsdangerous-signed httponly cookies** (`SECRET_KEY` from env, 8h lifetime).
- **Every route requires login** except `/login`, `/health`, and `/static/*`.
- Simple in-memory **login rate-limiting** (5 failures / 5 min per username+IP).
- One admin user is seeded from env; the schema is RBAC-ready (`users.is_admin`).

---

## SECURITY (read before exposing)

This app **enforces login but is not hardened for the open internet.** For a home server:

- **Bind to your LAN, not the public internet.** Compose binds `127.0.0.1:8000`. To reach it from
  other devices on your LAN, front it with a reverse proxy or change the bind to your LAN IP — but
  do **not** port-forward 8000 to the internet.
- **Put it behind HTTPS + a reverse proxy** (e.g. **Caddy** for automatic TLS, or nginx), **or a VPN**
  (WireGuard/Tailscale). Cookies are httponly but are only secure over HTTPS.
- **Set strong secrets** in `.env` (`SECRET_KEY`, `APP_ADMIN_PASSWORD`, `POSTGRES_PASSWORD`).
- **Never commit** `.env`, the DB, `models/`, or any uploads/imaging (all gitignored).
- Postgres is **not** published to the host — only the api service reaches it on the compose network.

Example Caddyfile (TLS termination in front of the app):

```
mora.example.lan {
    reverse_proxy 127.0.0.1:8000
}
```

---

## Layout

```
services/api/
  app/
    main.py            FastAPI app: routes, auth middleware, inference + experiment endpoints
    mora_engine.py     the live MoRA mechanism (loads persisted models; per-modality reliability)
    experiments.py     live experiment runners (reuse verified pipeline logic on cached features)
    upload.py          UPLOAD-mode radiomics extraction + clinical form handling
    auth.py            session cookies, bcrypt hashing, login rate-limiting, admin seeding
    db.py              SQLAlchemy models (User, Run) + SQLite/Postgres session
    templates/         Jinja2: base, login, dashboard, infer, experiments, history
  requirements.txt     web-service deps
  run_local.sh         local dev launcher (SQLite)
infrastructure/docker/
  Dockerfile           lean CPU image (sklearn-only; no torch at runtime)
  docker-compose.yml   api + postgres, read-only model/data mounts
  .env.example         copy to .env and fill in secrets
services/ai/califusion-cnn/
  scripts/build_deployment_models.py   persists the models the app loads
  models/                              built artifacts (gitignored)
```
