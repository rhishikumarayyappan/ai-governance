"""API routes for the AI system registry."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from governance.db.database import get_db
from governance.registry import service
from governance.registry.schemas import AISystemCreate, AISystemRead

router = APIRouter(prefix="/api/v1", tags=["registry"])


@router.get("/systems", response_model=list[AISystemRead])
def get_systems(db: Session = Depends(get_db)):
    """Return all registered AI systems."""
    return service.list_systems(db)


@router.post(
    "/systems", response_model=AISystemRead, status_code=status.HTTP_201_CREATED
)
def post_system(payload: AISystemCreate, db: Session = Depends(get_db)):
    """Register (create and persist) a new AI system."""
    return service.create_system(db, payload)
