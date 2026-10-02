"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent

# Load .env into os.environ so third-party SDKs (Anthropic, Google) can find their keys
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings."""

    # Project paths
    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # OCR Provider
    OCR_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_OCR_MODEL: str = "gemini-3.5-flash"

    # Solver web search
    # Upfront web lookup for every pop-culture-looking clue. Off by default: on
    # gauntlet puzzle 104 it didn't change the solved grid and doubled the cost.
    WEB_PREPASS_ENABLED: bool = False
    # Let the high-effort solve passes (4+) call web search when the model
    # decides it needs to, targeting only clues still unsolved by then.
    SOLVER_WEB_SEARCH: bool = True

    # Abuse limits (in-memory, per server process; see crosswise/api/rate_limit.py)
    MAX_UPLOAD_MB: int = 20
    RATE_LIMIT_ENABLED: bool = True
    # "solve" = /start-pipeline, /mask, /solve: each spends OCR and/or Claude calls
    RATE_LIMIT_SOLVES_PER_HOUR: int = 20  # per client IP
    RATE_LIMIT_SOLVES_PER_DAY: int = 200  # all clients combined
    # "upload" = /upload, /manual-crop: grid detection CPU and disk
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 60  # per client IP

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
