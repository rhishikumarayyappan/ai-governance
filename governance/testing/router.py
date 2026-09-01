"""HTTP API for the testing engine (Component 3).

Three endpoints:
    POST /api/v1/test-runs              -> upload model + data, run bias tests
    GET  /api/v1/test-runs/{run_id}     -> run status
    GET  /api/v1/test-runs/{run_id}/results -> the 5 metric results

All handlers are plain ``def`` (no async): uploads are read synchronously via
``UploadFile.file`` and ``run_bias_tests`` is synchronous end to end.
"""

import io
import pickle

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from governance.db.database import get_session
from governance.db.models import TestRun
from governance.testing.engine import get_run_results, run_bias_tests
from governance.testing.schemas import TestRunResponse, TestRunResultsResponse

router = APIRouter(prefix="/api/v1", tags=["testing"])

# 422 Unprocessable Content — literal to stay stable across Starlette renames.
_UNPROCESSABLE = 422


@router.post(
    "/test-runs",
    response_model=TestRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_test_run(
    system_id: str = Form(...),
    model_file: UploadFile = File(..., description="Serialised model file (.pkl)"),
    test_data: UploadFile = File(..., description="Test dataset (.csv)"),
    target_column: str = Form(
        ..., description="Name of the target/label column in the CSV"
    ),
    protected_attributes: str = Form(
        ...,
        description="Comma-separated list of protected attribute column names "
        "e.g. gender,race",
    ),
):
    """Upload a model and test dataset, run the full bias suite, persist results."""
    # 1. Parse the comma-separated protected attributes.
    attrs = [a.strip() for a in protected_attributes.split(",") if a.strip()]
    if not attrs:
        raise HTTPException(
            status_code=_UNPROCESSABLE, detail="protected_attributes cannot be empty"
        )

    # 2. Load the model straight from the uploaded bytes (no temp file).
    try:
        model = pickle.loads(model_file.file.read())
    except Exception:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="Could not load model file. Ensure it is a valid .pkl file.",
        )

    # 3. Parse the CSV into a DataFrame.
    try:
        df = pd.read_csv(io.BytesIO(test_data.file.read()))
    except Exception:
        raise HTTPException(
            status_code=_UNPROCESSABLE, detail="Could not parse CSV file."
        )

    # 4. The target column must exist.
    if target_column not in df.columns:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=(
                f"target_column '{target_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    # 5. Split features from labels (protected columns stay in X_test — the
    #    engine drops them itself before predicting).
    y_test = df[target_column].values
    X_test = df.drop(columns=[target_column])

    # 6. Every named protected attribute must be a real column.
    missing = [a for a in attrs if a not in X_test.columns]
    if missing:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail=f"Protected attribute columns not found in CSV: {missing}",
        )

    # 7-8. Run the engine. It records the TestRun (including "failed" state)
    #      before re-raising, so there is no DB cleanup to do here.
    try:
        run_id = run_bias_tests(
            system_id=system_id,
            model_source=model,
            X_test=X_test,
            y_test=y_test,
            protected_attributes=attrs,
            config={
                "model_filename": model_file.filename,
                "test_data_filename": test_data.filename,
                "target_column": target_column,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    # 9. Return the completed TestRun.
    with get_session() as session:
        run = session.get(TestRun, run_id)
        return TestRunResponse.model_validate(run)


@router.get("/test-runs/{run_id}", response_model=TestRunResponse)
def get_test_run(run_id: str):
    """Return the status record for one test run."""
    with get_session() as session:
        run = session.get(TestRun, run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test run {run_id} not found",
            )
        return TestRunResponse.model_validate(run)


@router.get("/test-runs/{run_id}/results", response_model=TestRunResultsResponse)
def get_test_run_results(run_id: str):
    """Return all metric results for one test run."""
    with get_session() as session:
        run = session.get(TestRun, run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test run {run_id} not found",
            )

    results = get_run_results(run_id)
    return TestRunResultsResponse(
        run_id=run_id, total_results=len(results), results=results
    )
