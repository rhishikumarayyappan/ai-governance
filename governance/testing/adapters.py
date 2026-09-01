"""Model adapter layer — Component 1 of the Phase 1 testing engine.

A universal translator so the testing engine can work with any model regardless
of how it is delivered: an in-memory sklearn estimator, a pickled file on disk,
or a REST endpoint.

Every adapter exposes the same interface:

    predict(X)        -> 1D numpy array of class predictions
    predict_proba(X)  -> 2D numpy array of class probabilities,
                         or None if the model cannot produce them
                         (this never raises — it returns None cleanly)

Use ``load_adapter(source)`` to get the right adapter for a given input.
"""

from __future__ import annotations

import abc
import json
import pickle
from pathlib import Path
from typing import Any
from urllib import request as _urlrequest
from urllib.error import URLError

import numpy as np

__all__ = [
    "ModelAdapter",
    "SklearnAdapter",
    "PickleAdapter",
    "APIAdapter",
    "load_adapter",
    "AdapterError",
]


class AdapterError(Exception):
    """Raised when an adapter cannot be built or a model cannot be loaded."""


def _to_feature_list(X: Any) -> list[list[Any]]:
    """Best-effort conversion of a feature matrix to plain nested lists (for JSON)."""
    if hasattr(X, "values"):  # pandas DataFrame
        return X.values.tolist()
    return np.asarray(X).tolist()


class ModelAdapter(abc.ABC):
    """Common interface every adapter implements."""

    @abc.abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return a 1D array with one prediction per row of X."""

    @abc.abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray | None:
        """Return a 2D array of class probabilities, or None if unavailable."""


class SklearnAdapter(ModelAdapter):
    """Wraps an already-fitted, in-memory sklearn-compatible estimator."""

    def __init__(self, model: Any):
        if not hasattr(model, "predict"):
            raise AdapterError(
                "SklearnAdapter needs a fitted estimator with a .predict() method; "
                f"got {type(model)!r}"
            )
        self.model = model

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray | None:
        proba_fn = getattr(self.model, "predict_proba", None)
        if proba_fn is None:
            return None
        try:
            proba = np.asarray(proba_fn(X))
        except Exception:
            # e.g. SVC(probability=False) exposes the method but raises on call.
            return None
        if proba.ndim != 2:
            return None
        return proba


class PickleAdapter(SklearnAdapter):
    """Loads a pickled sklearn model from disk, then behaves like SklearnAdapter.

    Because it delegates to SklearnAdapter once loaded, its predictions are
    identical to wrapping the same model in memory.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise AdapterError(f"PickleAdapter: no model file found at '{self.path}'")
        try:
            with self.path.open("rb") as fh:
                model = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - we re-raise as a clear error
            raise AdapterError(
                f"PickleAdapter: could not unpickle a model from '{self.path}': {exc}"
            ) from exc
        super().__init__(model)


class APIAdapter(ModelAdapter):
    """Calls a REST endpoint that returns predictions.

    Expected contract:
        POST {base_url}{predict_path}        body: {"instances": [[...], ...]}
            -> {"predictions": [...]}
        POST {base_url}{predict_proba_path}  body: {"instances": [[...], ...]}
            -> {"probabilities": [[...], ...]}   (optional; missing -> None)
    """

    def __init__(
        self,
        url: str,
        *,
        predict_path: str = "/predict",
        predict_proba_path: str = "/predict-proba",
        timeout: float = 30.0,
    ):
        self.base_url = url.rstrip("/")
        self.predict_path = predict_path
        self.predict_proba_path = predict_proba_path
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = _urlrequest.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError) as exc:
            raise AdapterError(
                f"APIAdapter: request to {self.base_url + path} failed: {exc}"
            ) from exc

    def predict(self, X: Any) -> np.ndarray:
        body = self._post(self.predict_path, {"instances": _to_feature_list(X)})
        return np.asarray(body["predictions"])

    def predict_proba(self, X: Any) -> np.ndarray | None:
        try:
            body = self._post(
                self.predict_proba_path, {"instances": _to_feature_list(X)}
            )
        except AdapterError:
            return None
        probs = body.get("probabilities")
        if probs is None:
            return None
        arr = np.asarray(probs)
        return arr if arr.ndim == 2 else None


def load_adapter(source: Any) -> ModelAdapter:
    """Return the appropriate adapter for ``source``.

    - an existing ModelAdapter                  -> returned unchanged
    - an ``http://`` / ``https://`` URL string  -> APIAdapter
    - a path to a file on disk (str or Path)    -> PickleAdapter
    - a fitted estimator with ``.predict``      -> SklearnAdapter
    """
    if isinstance(source, ModelAdapter):
        return source

    if isinstance(source, (str, Path)):
        text = str(source)
        if text.startswith(("http://", "https://")):
            return APIAdapter(text)
        return PickleAdapter(source)

    if hasattr(source, "predict"):
        return SklearnAdapter(source)

    raise AdapterError(
        f"load_adapter: cannot build an adapter from {type(source)!r}. "
        "Expected a fitted estimator, a .pkl path, or a URL."
    )
