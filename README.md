# AI Governance Platform

An AI regulatory compliance testing platform. Enterprises connect their AI
models; it runs technical tests for bias, fairness, and explainability, maps the
results to EU AI Act and GDPR obligations, and produces compliance dashboards and
audit-ready reports.

See [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) for the full architecture and phase
plan, and [`docs/PROGRESS.md`](docs/PROGRESS.md) for current status.

## Stack

Python 3.11 · Poetry · FastAPI · SQLite (SQLAlchemy) · Streamlit · fairlearn /
shap / lime · ReportLab · pytest

## Local setup

```bash
pyenv install 3.11.9
pyenv local 3.11.9
poetry install
```

## Run the API

```bash
poetry run uvicorn governance.main:app --reload
```

- `GET  /health` — service health
- `GET  /api/v1/systems` — list registered AI systems
- `POST /api/v1/systems` — register a new AI system
- Interactive docs: http://127.0.0.1:8000/docs

## Tests

```bash
poetry run pytest
```
