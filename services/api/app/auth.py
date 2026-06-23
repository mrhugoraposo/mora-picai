"""
auth.py — session-cookie auth, password hashing, login rate-limiting.

Sessions are itsdangerous-signed cookies (httponly). Passwords hashed with passlib bcrypt.
An admin user is seeded from APP_ADMIN_USER / APP_ADMIN_PASSWORD on startup. Every route
requires login except /login, /health and static assets (enforced in main.py).
"""
from __future__ import annotations
import os
import time

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from passlib.context import CryptContext

from .db import SessionLocal, User

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")
COOKIE_NAME = "mora_session"
SESSION_MAX_AGE = int(os.environ.get("SESSION_MAX_AGE", str(8 * 3600)))  # 8h
_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="mora-session")
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- simple in-memory login rate limiting (per-username + per-IP) ---
_FAILS: dict[str, list[float]] = {}
_MAX_FAILS = 5
_WINDOW = 300.0  # 5 min


def hash_password(pw: str) -> str:
    return _pwd.hash(pw)


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return _pwd.verify(pw, pw_hash)
    except Exception:
        return False


def rate_limited(key: str) -> bool:
    now = time.time()
    hist = [t for t in _FAILS.get(key, []) if now - t < _WINDOW]
    _FAILS[key] = hist
    return len(hist) >= _MAX_FAILS


def record_fail(key: str):
    _FAILS.setdefault(key, []).append(time.time())


def clear_fails(key: str):
    _FAILS.pop(key, None)


def make_session_cookie(user: User) -> str:
    return _serializer.dumps({"uid": user.id, "u": user.username, "admin": user.is_admin})


def read_session_cookie(token: str):
    if not token:
        return None
    try:
        return _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def authenticate(username: str, password: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and verify_password(password, user.password_hash):
            return user
        return None
    finally:
        db.close()


def seed_admin():
    """Create/refresh the admin user from env. Idempotent."""
    user = os.environ.get("APP_ADMIN_USER", "admin")
    pw = os.environ.get("APP_ADMIN_PASSWORD", "")
    if not pw:
        pw = "changeme"  # dev fallback; README + .env.example tell prod to set a real one
        _seed_warning[0] = True
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == user).first()
        if existing:
            existing.password_hash = hash_password(pw)
            existing.is_admin = True
        else:
            db.add(User(username=user, password_hash=hash_password(pw), is_admin=True))
        db.commit()
        return user
    finally:
        db.close()


_seed_warning = [False]  # set when a default password was used


def used_default_password() -> bool:
    return _seed_warning[0]
