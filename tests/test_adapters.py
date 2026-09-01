"""Phase 1, Week 1 — tests for the model adapter layer.

Five tests, as specified in docs/BUILD_PLAN.md (Component 1 / Required pytest Tests).
"""

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression

from governance.testing.adapters import (
    AdapterError,
    APIAdapter,
    PickleAdapter,
    SklearnAdapter,
    load_adapter,
)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=42
    )
    return X, y


@pytest.fixture(scope="module")
def clf(data):
    X, y = data
    return LogisticRegression(max_iter=1000).fit(X, y)


def test_sklearn_predict_returns_correct_shape(clf, data):
    X, _ = data
    preds = SklearnAdapter(clf).predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)  # one prediction per input row


def test_pickle_adapter_matches_sklearn_adapter(clf, data, tmp_path):
    X, _ = data
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    in_memory = SklearnAdapter(clf).predict(X)
    from_disk = PickleAdapter(model_path).predict(X)

    assert np.array_equal(in_memory, from_disk)


def test_predict_proba_is_2d_array_or_none_never_raises(clf, data):
    X, y = data

    # Classifier that supports probabilities -> 2D array.
    proba = SklearnAdapter(clf).predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.ndim == 2
    assert proba.shape[0] == X.shape[0]

    # Regressor with no predict_proba at all -> None, cleanly, no exception.
    reg = LinearRegression().fit(X, y)
    assert SklearnAdapter(reg).predict_proba(X) is None


def test_load_adapter_returns_right_type_for_each_input(clf, tmp_path):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    assert isinstance(load_adapter(clf), SklearnAdapter)
    assert isinstance(load_adapter(model_path), PickleAdapter)
    assert isinstance(load_adapter(str(model_path)), PickleAdapter)
    assert isinstance(load_adapter("https://models.example.com/score"), APIAdapter)


def test_pickle_adapter_raises_clear_error_for_bad_path(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"
    with pytest.raises(AdapterError) as excinfo:
        PickleAdapter(missing)
    assert str(missing) in str(excinfo.value)
