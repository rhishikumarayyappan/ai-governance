"""FastAPI entry point for the AI Governance Platform.

Endpoints:
    GET  /health                            -> {"status": "ok", "version": ...}
    GET  /api/v1/systems                    -> list all registered AI systems
    POST /api/v1/systems                    -> register a new AI system
    POST /api/v1/test-runs                  -> upload model + data, run bias tests
    GET  /api/v1/test-runs/{run_id}         -> test run status
    GET  /api/v1/test-runs/{run_id}/results -> test run metric results
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from governance.config import settings
from governance.db.database import init_db
from governance.registry.router import router as registry_router
from governance.testing.router import router as testing_router

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
app.include_router(testing_router)
