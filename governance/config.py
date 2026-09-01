"""Application configuration.

Values can be overridden with environment variables (prefix ``AIGOV_``) or a
``.env`` file in the project root. See ``.env.example``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: .../ai-governance/
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AIGOV_",
        extra="ignore",
    )

    app_name: str = "AI Governance Platform"
    version: str = "0.1.0"

    # SQLite database file, created automatically on first run.
    database_url: str = f"sqlite:///{BASE_DIR / 'ai_governance.db'}"

    # Origins allowed to call the API from a browser (Streamlit dashboard, etc.).
    cors_origins: list[str] = [
        "http://localhost:8501",
        "http://localhost:8502",
        "http://127.0.0.1:8501",
    ]


settings = Settings()
