"""Request/response models for the AI system registry."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from governance.db.models import RiskTier


class AISystemCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    model_type: str | None = Field(
        default=None, description="classification / regression / llm / ranking"
    )
    risk_tier: RiskTier | None = None
    sector: str | None = Field(
        default=None, description="finance / healthcare / hr / insurance / other"
    )
    owner: str | None = Field(default=None, description="owner email address")


class AISystemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    model_type: str | None
    risk_tier: RiskTier | None
    sector: str | None
    owner: str | None
    created_at: datetime
