"""Runtime configuration for the Payroll Mapping Engine."""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeEnvironment(str, Enum):
    """Deployment environment names understood by the service."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    AZURE = "azure"


class DatasetSource(str, Enum):
    """Supported dataset locations."""

    LOCAL = "local"
    AZURE = "azure"


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
    environment: RuntimeEnvironment = Field(default=RuntimeEnvironment.DEVELOPMENT)

    dataset_source: DatasetSource | None = Field(default=None)
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
    openai_max_tokens: int = Field(default=128, ge=1)
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_timeout_seconds: float = Field(default=20.0, gt=0.0)
    openai_max_retries: int = Field(default=1, ge=0)
    gpt_adjudication_enabled: bool = Field(default=False)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-01")
    azure_openai_deployment: str | None = Field(default=None)

    default_mode: PrecedenceMode = Field(default=PrecedenceMode.MAX_OCCURRENCE)
    tie_break_strategy: TieBreakStrategy = Field(default=TieBreakStrategy.FIRST_SEEN)
    date_format: str = Field(default="%m/%d/%Y")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_reload: bool = Field(default=False)
    api_workers: int = Field(default=1, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_json: bool = Field(default=False)

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: str | RuntimeEnvironment) -> str | RuntimeEnvironment:
        if isinstance(value, RuntimeEnvironment):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "dev": RuntimeEnvironment.DEVELOPMENT.value,
            "prod": RuntimeEnvironment.PRODUCTION.value,
        }
        return aliases.get(normalized, normalized)

    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment in {RuntimeEnvironment.PRODUCTION, RuntimeEnvironment.AZURE}

    @property
    def uses_azure_storage(self) -> bool:
        return bool(self.azure_storage_connection_string)

    @property
    def effective_dataset_source(self) -> DatasetSource:
        if self.dataset_source is not None:
            return self.dataset_source
        return DatasetSource.AZURE if self.uses_azure_storage else DatasetSource.LOCAL

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


_RESERVED_LOG_RECORD_KEYS = set(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
)


class JsonLogFormatter(logging.Formatter):
    """Format log records as compact JSON including structured extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging() -> None:
    """Configure process logging once at startup."""

    level = getattr(logging, settings.log_level.value)
    if settings.log_json:
        formatter: logging.Formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
