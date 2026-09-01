"""Response models for the testing API (Component 3).

All three models set ``from_attributes=True`` so they can be built directly from
SQLAlchemy model instances via ``Model.model_validate(orm_obj)``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TestResultItem(BaseModel):
    """One metric result, as returned inside TestRunResultsResponse."""

    model_config = ConfigDict(from_attributes=True)

    metric_name: str
    metric_value: float
    threshold: float
    status: str
    module: str
    detail: dict


class TestRunResponse(BaseModel):
    """A single test run's status record."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # The ORM column is ``id``; expose it as ``run_id``.
    run_id: str = Field(validation_alias="id")
    system_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None


class TestRunResultsResponse(BaseModel):
    """All metric results for one test run."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    total_results: int
    results: list[dict]
