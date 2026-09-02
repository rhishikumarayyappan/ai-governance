"""One-off, idempotent migration: add the Phase 2 statistical columns to
``test_results``.

There is no Alembic yet (BUILD_PLAN puts it at Phase 7). ``Base.metadata.create_all``
creates *missing tables* but never alters existing ones, so a database that
predates the Phase 2 wiring sub-step will not gain the new ``test_results``
columns on its own. This script adds them with ``ALTER TABLE ... ADD COLUMN``.

Safe to run any number of times: it checks ``PRAGMA table_info`` first and only
adds columns that are missing. The tests do NOT need this — ``conftest.py``
builds fresh tables per test.

Usage:
    poetry run python scripts/migrate_add_testresult_stats.py
    poetry run python scripts/migrate_add_testresult_stats.py path/to/other.db
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# column name -> SQLite column type (matching SQLAlchemy's rendering).
# All are nullable with no default; existing rows get NULL.
_NEW_COLUMNS: dict[str, str] = {
    "confidence_interval_lower": "FLOAT",
    "confidence_interval_upper": "FLOAT",
    "p_value": "FLOAT",
    "corrected_threshold": "FLOAT",
    "correction_method": "VARCHAR",
    "reliability_tier": "VARCHAR",
    "sample_size": "INTEGER",
}


def _default_db_path() -> Path:
    """The database the app is configured to use."""
    from governance.config import settings

    url = settings.database_url
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise SystemExit(f"This script only handles SQLite URLs; got {url!r}")
    return Path(url[len(prefix):])


def migrate(db_path: str | Path) -> list[str]:
    """Add any missing Phase 2 columns to ``test_results`` in ``db_path``.

    Returns the list of column names actually added (empty if already current).
    Raises if ``test_results`` does not exist.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise SystemExit(f"No database file at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "test_results" not in tables:
            raise SystemExit(
                f"{db_path} has no 'test_results' table — run the app once to "
                f"create the schema, then re-run this migration."
            )

        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(test_results)")
        }
        added: list[str] = []
        for name, col_type in _NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE test_results ADD COLUMN {name} {col_type}"
                )
                added.append(name)
        conn.commit()
        return added
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    db_path = Path(argv[1]) if len(argv) > 1 else _default_db_path()
    added = migrate(db_path)
    if added:
        print(f"{db_path}: added {len(added)} column(s): {', '.join(added)}")
    else:
        print(f"{db_path}: already up to date, nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
