"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # OCR Provider
    OCR_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_OCR_MODEL: str = "gemini-3-flash-preview"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # or "text"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
