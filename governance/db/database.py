"""SQLite database connection, session factory, and table creation.

- The SQLite file is created automatically the first time the engine connects.
- ``SessionLocal`` is the single session factory used across the application.
- WAL journal mode is enabled on every connection for better concurrent access
  (see Risk #4 in docs/BUILD_PLAN.md).
"""

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from governance.config import settings

# check_same_thread=False: FastAPI runs synchronous endpoints in a threadpool,
# so a connection may be used from a different thread than it was created on.
# This is safe here because each request uses its own short-lived session.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable WAL mode and foreign-key enforcement on each new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_db():
    """FastAPI dependency that yields a database session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_session():
    """Context manager yielding a database session for non-API code.

    Used by the testing engine, dashboard and SDK — anything that is not a
    FastAPI request handler. Does NOT commit automatically: the caller is
    responsible for calling ``session.commit()``. The session is always closed
    on exit.

    ``SessionLocal`` is looked up on each call, so tests can redirect the
    database by monkeypatching ``governance.db.database.SessionLocal``.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables. Safe to call on every startup — existing tables are left alone."""
    # Import models so they are registered on Base.metadata before create_all.
    from governance.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
