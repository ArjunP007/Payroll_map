"""Pydantic models for the Payroll Mapping Engine."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import DatasetSource, PrecedenceMode


class RawCandidateRecord(BaseModel):
    """One raw candidate object from the nested source JSON."""

    model_config = ConfigDict(extra="ignore")

    internalCode: str = Field(min_length=1)
    internalDescription: str = Field(default="")
    LastModifiedDate: str = Field(min_length=1)


class NormalizedRecord(BaseModel):
    """Flat, computation-ready mapping record."""

    model_config = ConfigDict(frozen=True)

    priorCode: str
    internalCode: str
    lastModifiedDate: datetime
    candidateIndex: int = Field(ge=0)
    globalIndex: int = Field(ge=0)


class MappingResult(BaseModel):
    """The only item shape returned to API callers."""

    model_config = ConfigDict(extra="forbid")

    priorCode: str
    internalCode: str


class MappingDecisionDetail(BaseModel):
    """Internal audit detail. Never returned from public mapping endpoints."""

    priorCode: str
    winningCode: str
    mode: PrecedenceMode
    primaryRule: str
    secondaryRule: str | None = None
    candidateCount: int
    uniqueCandidates: list[str]
    tiedCandidates: list[str]
    occurrenceCounts: dict[str, int]
    latestDates: dict[str, str]
    tieBreakApplied: bool = False
    tieBreakStrategy: str | None = None
    gptAdjudicated: bool = False
    gptRawResponse: str | None = None


class MappingRequest(BaseModel):
    """Batch mapping request. The caller selects only the precedence mode."""

    model_config = ConfigDict(extra="forbid")

    mode: PrecedenceMode

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str | PrecedenceMode) -> str | PrecedenceMode:
        if isinstance(value, str):
            return value.strip().upper()
        return value


class ReloadRequest(BaseModel):
    """Optional request body for dataset reload."""

    model_config = ConfigDict(extra="forbid")

    source: DatasetSource | None = Field(default=None)

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, value: str | DatasetSource | None) -> str | DatasetSource | None:
        if value is None:
            return None
        if isinstance(value, DatasetSource):
            return value
        normalized = str(value).strip().lower()
        allowed = {item.value for item in DatasetSource}
        if normalized not in allowed:
            raise ValueError(f"source must be one of {sorted(allowed)}")
        return normalized


class HealthResponse(BaseModel):
    """Readiness and liveness response."""

    status: str
    appName: str
    version: str
    environment: str
    datasetLoaded: bool
    priorCodeCount: int
    recordCount: int


class ReloadResponse(BaseModel):
    """Dataset reload response."""

    status: str
    priorCodeCount: int
    recordCount: int
    message: str


class PriorCodesResponse(BaseModel):
    """Admin response containing known prior codes in source order."""

    totalPriorCodes: int
    priorCodes: list[str]


class ErrorResponse(BaseModel):
    """Standard sanitized error envelope."""

    error: str
    detail: str | None = None
    statusCode: int
