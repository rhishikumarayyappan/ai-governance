"""Phase 1, Week 2, Component 3 — tests for the testing API endpoints.

Uses FastAPI's TestClient and the function-scoped `test_db` fixture from
conftest.py. The FastAPI app is imported lazily inside the `client` fixture so
that `main.py`'s module-level `init_db()` targets the temp database, never the
real `ai_governance.db`.
"""

import pickle
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from governance.db.database import get_session
from governance.db.models import AISystem, RiskTier


@pytest.fixture
def client(test_db):
    from governance.main import app  # lazy: after test_db has patched the DB

    return TestClient(app)


@pytest.fixture
def system_id(test_db) -> str:
    with get_session() as session:
        system = AISystem(
            name="api-test-system",
            model_type="classification",
            risk_tier=RiskTier.high,
            sector="finance",
            owner="test@example.com",
        )
        session.add(system)
        session.commit()
        return system.id


@pytest.fixture
def payload():
    """A matching (csv_bytes, model_bytes) pair.

    CSV: feature_1 (float), gender (M/F), target (0/1), 24 rows, balanced target.
    Model: LogisticRegression trained on feature_1 only (the engine drops
    'gender' before predicting, leaving one feature column).
    """
    rng = np.random.RandomState(7)
    n = 24
    feature_1 = rng.randn(n)
    gender = rng.choice(["M", "F"], size=n)
    target = (feature_1 > np.median(feature_1)).astype(int)
    df = pd.DataFrame({"feature_1": feature_1, "gender": gender, "target": target})

    model = LogisticRegression().fit(df[["feature_1"]].values, target)

    return df.to_csv(index=False).encode(), pickle.dumps(model)


def _post_test_run(client, system_id, payload, *, target_column="target",
                   protected_attributes="gender"):
    csv_bytes, model_bytes = payload
    return client.post(
        "/api/v1/test-runs",
        data={
            "system_id": system_id,
            "target_column": target_column,
            "protected_attributes": protected_attributes,
        },
        files={
            "model_file": ("model.pkl", model_bytes, "application/octet-stream"),
            "test_data": ("data.csv", csv_bytes, "text/csv"),
        },
    )


@pytest.fixture
def created_run(client, system_id, payload) -> dict:
    resp = _post_test_run(client, system_id, payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #

def test_successful_test_run_via_api(client, system_id, payload):
    resp = _post_test_run(client, system_id, payload)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "complete"
    uuid.UUID(body["run_id"])  # raises ValueError if not a valid UUID string


def test_invalid_system_id_returns_404(client, payload):
    fake_id = str(uuid.uuid4())
    resp = _post_test_run(client, fake_id, payload)

    assert resp.status_code == 404
    assert fake_id in resp.json()["detail"]


def test_bad_target_column_returns_422(client, system_id, payload):
    resp = _post_test_run(client, system_id, payload, target_column="not_a_column")

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "not_a_column" in detail
    assert "feature_1" in detail  # available columns are listed


def test_get_run_status(client, created_run):
    run_id = created_run["run_id"]
    resp = client.get(f"/api/v1/test-runs/{run_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["system_id"]
    assert body["status"] == "complete"


def test_get_run_results(client, created_run):
    run_id = created_run["run_id"]
    resp = client.get(f"/api/v1/test-runs/{run_id}/results")

    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert body["total_results"] == 5
    assert len(body["results"]) == 5
    for item in body["results"]:
        assert {"metric_name", "metric_value", "threshold", "status"} <= item.keys()
        assert item["status"] in ("pass", "warn", "fail")


def test_get_nonexistent_run_returns_404(client):
    resp = client.get("/api/v1/test-runs/fake-id-that-does-not-exist")

    assert resp.status_code == 404
