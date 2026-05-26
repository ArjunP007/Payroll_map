"""Runtime configuration for the Payroll Mapping Engine."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrecedenceMode(str, Enum):
    """Supported batch mapping modes."""

    ONE_TO_ONE = "ONE_TO_ONE"
    MAX_OCCURRENCE = "MAX_OCCURRENCE"
    LAST_MODIFIED_DATE = "LAST_MODIFIED_DATE"


class TieBreakStrategy(str, Enum):
    """Stable backend-only tie-break strategies."""

    FIRST_SEEN = "first_seen"
    LEXICOGRAPHIC = "lexicographic"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Environment-backed settings used by all application layers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Payroll Mapping Engine")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")

    dataset_local_path: Path = Field(
        default=Path("data/FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json")
    )
    azure_storage_connection_string: str | None = Field(default=None)
    azure_storage_container_name: str = Field(default="payroll-datasets")
    azure_storage_blob_name: str = Field(
        default="FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json"
    )
    azure_key_vault_url: str | None = Field(default=None)

    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    openai_max_tokens: int = Field(default=128)
    openai_temperature: float = Field(default=0.0)
    gpt_adjudication_enabled: bool = Field(default=False)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-01")
    azure_openai_deployment: str | None = Field(default=None)

    default_mode: PrecedenceMode = Field(default=PrecedenceMode.MAX_OCCURRENCE)
    tie_break_strategy: TieBreakStrategy = Field(default=TieBreakStrategy.FIRST_SEEN)
    date_format: str = Field(default="%m/%d/%Y")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    api_workers: int = Field(default=1)
    cors_origins: list[str] = Field(default=["*"])

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_json: bool = Field(default=False)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"development", "staging", "production"}
        if normalized not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return normalized

    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def uses_azure_openai(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_deployment)

    @property
    def effective_openai_model(self) -> str:
        return self.azure_openai_deployment or self.openai_model

    def refresh_runtime_secrets_from_env(self) -> None:
        """Refresh mutable secret fields after Key Vault populates env vars."""

        self.openai_api_key = os.environ.get("OPENAI_API_KEY", self.openai_api_key)
        self.azure_storage_connection_string = os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING",
            self.azure_storage_connection_string,
        )


settings = Settings()


def configure_logging() -> None:
    """Configure process logging once at startup."""

    level = getattr(logging, settings.log_level.value)
    if settings.log_json:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%dT%H:%M:%S")
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
