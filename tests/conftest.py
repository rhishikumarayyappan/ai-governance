"""Shared pytest fixtures.

`test_db` gives each test a brand-new, isolated SQLite database. The real
`ai_governance.db` is never touched by the test suite.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import governance.db.models  # noqa: F401 - registers tables on Base.metadata
from governance.db import database
from governance.db.database import Base


@pytest.fixture(scope="function")
def test_db(tmp_path, monkeypatch):
    """Function-scoped isolated database.

    Every test gets its own SQLite file, its own engine, freshly created tables,
    and zero state from any other test. `governance.db.database.SessionLocal` is
    monkeypatched to point at this engine, so `get_session()` (which looks the
    factory up on each call) and everything built on it use the temp database.
    """
    db_path = tmp_path / "test.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(test_engine, "connect")
    def _enable_fk(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    test_session_factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )

    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "SessionLocal", test_session_factory)

    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        test_engine.dispose()
