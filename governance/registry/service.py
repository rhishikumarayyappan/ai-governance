"""Business logic for the AI system registry.

Kept separate from the router so it can be reused (SDK, dashboard, tests)
without going through HTTP.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from governance.db.models import AISystem
from governance.registry.schemas import AISystemCreate


def list_systems(db: Session) -> list[AISystem]:
    return list(db.scalars(select(AISystem).order_by(AISystem.created_at)))


def create_system(db: Session, data: AISystemCreate) -> AISystem:
    system = AISystem(**data.model_dump())
    db.add(system)
    db.commit()
    db.refresh(system)
    return system
