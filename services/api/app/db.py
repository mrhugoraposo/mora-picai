"""
db.py — SQLAlchemy models + session for the MoRA console.

SQLite by default (local/dev, zero-config smoke testing). When DATABASE_URL is set
(e.g. postgresql+psycopg2://... in Docker) that backend is used instead. Stores users
and run history (who ran what, when, result summary) for auditability.
"""
from __future__ import annotations
import datetime as dt
import json
import os

from sqlalchemy import (create_engine, Column, Integer, String, DateTime, Text,
                        Boolean, ForeignKey)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SQLITE = "sqlite:///" + os.path.join(_HERE, "..", "mora_console.db")
DATABASE_URL = os.environ.get("DATABASE_URL", _DEFAULT_SQLITE)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    runs = relationship("Run", back_populates="user")


class Run(Base):
    """One inference or experiment execution — the audit trail."""
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    kind = Column(String(32), nullable=False)          # "inference" | "experiment"
    dataset = Column(String(32))                        # picai | lung1
    name = Column(String(128))                          # case id / experiment name
    summary = Column(Text)                              # JSON blob of the key result
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    user = relationship("User", back_populates="runs")

    def summary_obj(self):
        try:
            return json.loads(self.summary) if self.summary else {}
        except Exception:
            return {}


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def record_run(db, user_id, kind, dataset, name, summary: dict):
    run = Run(user_id=user_id, kind=kind, dataset=dataset, name=name,
              summary=json.dumps(summary)[:8000])
    db.add(run)
    db.commit()
    return run
