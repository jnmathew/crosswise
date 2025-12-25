"""
Configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    EXAMPLES_DIR: Path = DATA_DIR / "examples"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    CACHE_DIR: Path = DATA_DIR / "cache"
    LOGS_DIR: Path = DATA_DIR / "logs"

    # OCR Configuration
    TESSERACT_CMD: Optional[str] = None  # Auto-detect if None
    OCR_DPI: int = 350
    DEFAULT_DPI: int = 350
    TARGET_X_HEIGHT: int = 14
    MAX_UPSCALE: float = 3.0
    OCR_CONFIDENCE_MIN: int = 45  # Minimum confidence threshold (0-100)

    # Grid Detection
    MIN_GRID_SIZE: int = 5
    MAX_GRID_SIZE: int = 25
    EXPECTED_GRID_ASPECT_RATIO: float = 1.0
    GRID_ASPECT_TOLERANCE: float = 0.3

    # Clue OCR
    MIN_CLUE_CONFIDENCE: float = 0.6
    CLUE_PSM_MODE: int = 6  # Tesseract PSM for clues
    DIGIT_PSM_MODE: int = 11  # Tesseract PSM for cell numbers

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.2

    # API Configuration (Phase 3)
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # or "text"
    SAVE_DEBUG_IMAGES: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for dir_path in [
            self.DATA_DIR,
            self.EXAMPLES_DIR,
            self.OUTPUT_DIR,
            self.CACHE_DIR,
            self.LOGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
