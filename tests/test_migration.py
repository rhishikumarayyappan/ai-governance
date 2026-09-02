"""Tests for scripts/migrate_add_testresult_stats.py.

This script runs raw ALTER TABLE outside the ORM's create_all()/session path,
against what will eventually be real data. The three things that matter:
it does what it claims, running it twice is safe, and it never touches
existing data.
"""

import sqlite3

import pytest

from scripts.migrate_add_testresult_stats import _NEW_COLUMNS, migrate

_OLD_SCHEMA = """
CREATE TABLE test_runs (id VARCHAR PRIMARY KEY);
CREATE TABLE test_results (
    id VARCHAR PRIMARY KEY, run_id VARCHAR, module VARCHAR, metric_name VARCHAR,
    metric_value FLOAT, threshold FLOAT, status VARCHAR, detail JSON
);
INSERT INTO test_runs VALUES ('r1');
INSERT INTO test_results VALUES
    ('t1','r1','bias','demographic_parity_difference',0.1745,0.10,'fail','{"a":1}'),
    ('t2','r1','bias','overall_accuracy_floor',0.84,0.80,'pass','{}');
"""


def _make_old_db(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    conn.commit()
    conn.close()
    return path


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        return {
            r["id"]: dict(r)
            for r in conn.execute(
                "SELECT id, run_id, module, metric_name, metric_value, "
                "threshold, status, detail FROM test_results ORDER BY id"
            )
        }
    finally:
        conn.close()


def _columns(path):
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(test_results)")]
    finally:
        conn.close()


def test_migration_adds_the_new_columns(tmp_path):
    path = _make_old_db(tmp_path)
    added = migrate(path)

    assert set(added) == set(_NEW_COLUMNS)
    cols = _columns(path)
    for name in _NEW_COLUMNS:
        assert name in cols
    # the new columns are NULL on rows that predate them
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT p_value, corrected_threshold, correction_method, "
        "reliability_tier, sample_size, confidence_interval_lower, "
        "confidence_interval_upper FROM test_results WHERE id='t1'"
    ).fetchone()
    conn.close()
    assert row == (None,) * 7


def test_migration_is_idempotent_and_a_true_noop_when_current(tmp_path):
    path = _make_old_db(tmp_path)
    migrate(path)                       # bring it current
    before = _rows(path)
    before_cols = _columns(path)

    added_again = migrate(path)         # run again on an already-current DB

    assert added_again == []            # nothing added
    assert _columns(path) == before_cols
    # every pre-existing value byte-identical before and after the no-op run
    assert _rows(path) == before


def test_migration_preserves_existing_data(tmp_path):
    path = _make_old_db(tmp_path)
    before = _rows(path)

    migrate(path)

    after = _rows(path)
    assert after == before             # the original 8 columns, both rows, unchanged


def test_migration_refuses_a_db_without_test_results(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()       # a file, but no schema
    with pytest.raises(SystemExit, match="test_results"):
        migrate(path)
