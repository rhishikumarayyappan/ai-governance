"""ORM table definitions.

Four tables, exactly as specified in docs/BUILD_PLAN.md:
    ai_systems, test_runs, test_results, compliance_scores

Design notes:
- Primary keys are string UUIDs (generated app-side) so records can be created
  and referenced without a round-trip to the database.
- ``risk_tier`` is the one field the build plan marks as an enum, so it is a real
  Python/SQL enum. The other category fields (model_type, sector, status,
  regulation, module) are stored as strings per the build plan; their allowed
  values are documented inline and enforced at the API layer.
- ``config``, ``detail`` and ``evidence`` are JSON columns.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from governance.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RiskTier(str, enum.Enum):
    """EU AI Act risk classification."""

    unacceptable = "unacceptable"
    high = "high"
    limited = "limited"
    minimal = "minimal"


class AISystem(Base):
    __tablename__ = "ai_systems"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # classification / regression / llm / ranking
    model_type: Mapped[str | None] = mapped_column(String, nullable=True)
    risk_tier: Mapped[RiskTier | None] = mapped_column(
        Enum(RiskTier, native_enum=False, length=20), nullable=True
    )
    # finance / healthcare / hr / insurance / other
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    owner: Mapped[str | None] = mapped_column(String, nullable=True)  # email
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    test_runs: Mapped[list["TestRun"]] = relationship(
        back_populates="system",
        cascade="all, delete-orphan",
        order_by="TestRun.started_at",
    )


class TestRun(Base):
    __tablename__ = "test_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    system_id: Mapped[str] = mapped_column(
        ForeignKey("ai_systems.id", ondelete="CASCADE"), nullable=False
    )
    # pending / running / complete / failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    system: Mapped["AISystem"] = relationship(back_populates="test_runs")
    results: Mapped[list["TestResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    compliance_scores: Mapped[list["ComplianceScore"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class TestResult(Base):
    __tablename__ = "test_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False
    )
    # bias / explainability / robustness
    module: Mapped[str] = mapped_column(String, nullable=False)
    # demographic_parity_difference / equalized_odds / ...
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    # the effective threshold this result was judged against (a run's
    # per-system override if one exists, else the BiasTestSuite default)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # pass / warn / fail / indeterminate
    #   "indeterminate" is set by the engine (never by BiasTestSuite) when
    #   reliability scoring returns tier="insufficient_data" — the data cannot
    #   support any verdict. It is the ABSENCE of a result, not a soft "warn".
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ---- Phase 2 statistical layer (all nullable; pre-Phase-2 rows have none) ----
    confidence_interval_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    # p_value / corrected_threshold / correction_method are populated ONLY for
    # the 4 group-comparison fairness metrics. overall_accuracy_floor leaves
    # them NULL: shuffling group labels cannot change overall accuracy, so a
    # permutation p-value for it is structurally meaningless — there is no
    # group-comparison hypothesis to test significance against.
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    correction_method: Mapped[str | None] = mapped_column(String, nullable=True)
    # "reliable" / "unstable" / "insufficient_data"
    reliability_tier: Mapped[str | None] = mapped_column(String, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    run: Mapped["TestRun"] = relationship(back_populates="results")


class ComplianceScore(Base):
    __tablename__ = "compliance_scores"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False
    )
    # eu_ai_act / gdpr
    regulation: Mapped[str] = mapped_column(String, nullable=False)
    # "Article 9" / "Article 10" / ...
    article: Mapped[str] = mapped_column(String, nullable=False)
    # compliant / partial / non_compliant / not_applicable
    status: Mapped[str] = mapped_column(String, nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    run: Mapped["TestRun"] = relationship(back_populates="compliance_scores")
