#!/usr/bin/env bash
# Local dev launcher for the MoRA Research Console (SQLite, no Docker).
# Usage: ./run_local.sh [port]   (defaults to 8080)
set -e
cd "$(dirname "$0")"

PORT="${1:-8080}"
VENV_PY="../ai/califusion-cnn/.venv/bin/python"

export SECRET_KEY="${SECRET_KEY:-dev-local-secret-change-me}"
export APP_ADMIN_USER="${APP_ADMIN_USER:-admin}"
export APP_ADMIN_PASSWORD="${APP_ADMIN_PASSWORD:-changeme}"
# DATABASE_URL unset -> SQLite (services/api/mora_console.db)

echo "MoRA Console -> http://127.0.0.1:${PORT}  (admin user: ${APP_ADMIN_USER})"
exec "$VENV_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}" --reload
