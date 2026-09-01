"""FastAPI entry point for the AI Governance Platform.

Phase 0 scope (see docs/BUILD_PLAN.md):
    GET  /health           -> {"status": "ok", "version": "0.1.0"}
    GET  /api/v1/systems    -> list all registered AI systems
    POST /api/v1/systems    -> register a new AI system
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from governance.config import settings
from governance.db.database import init_db
from governance.registry.router import router as registry_router

# Create the SQLite file and all tables when the app is loaded (synchronous,
# runs once at startup). create_all() is a no-op if the tables already exist.
init_db()

app = FastAPI(title=settings.app_name, version=settings.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "version": settings.version}


app.include_router(registry_router)
