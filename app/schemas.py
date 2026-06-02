"""Pydantic models for the Payroll Mapping Engine."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import PAYROLL_CATEGORIES, DatasetSource, PayrollCategory, PrecedenceMode


Metadata = Mapping[str, Any]


class RawCandidateRecord(BaseModel):
    """One raw candidate object from the nested source JSON."""

    model_config = ConfigDict(extra="allow")

    globalCode: str = Field(min_length=1)
    LastModifiedDate: str = Field(min_length=1)


class NormalizedRecord(BaseModel):
    """Flat, computation-ready mapping record."""

    model_config = ConfigDict(frozen=True)

    category: PayrollCategory
    priorCode: str
    globalCode: str
    lastModifiedDate: datetime
    candidateIndex: int = Field(ge=0)
    globalIndex: int = Field(ge=0)
    metadata: Metadata = Field(default_factory=dict)


class MappingResult(BaseModel):
    """The only item shape returned to API callers."""

    model_config = ConfigDict(extra="forbid")

    priorCode: str
    globalCode: str


class CategoryMappingResponse(BaseModel):
    """Public EDT-grouped mapping response."""

    model_config = ConfigDict(extra="forbid")

    Earnings: list[MappingResult] = Field(default_factory=list)
    Deductions: list[MappingResult] = Field(default_factory=list)
    Taxes: list[MappingResult] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> "CategoryMappingResponse":
        return cls()

    @classmethod
    def from_category_map(
        cls,
        values: Mapping[PayrollCategory, list[MappingResult]],
    ) -> "CategoryMappingResponse":
        payload = {category.value: list(values.get(category, [])) for category in PAYROLL_CATEGORIES}
        return cls.model_validate(payload)

    def as_category_map(self) -> dict[PayrollCategory, list[MappingResult]]:
        return {
            category: list(getattr(self, category.value))
            for category in PAYROLL_CATEGORIES
        }


class MappingDecisionDetail(BaseModel):
    """Mapping audit detail. Never returned from public mapping endpoints."""

    category: PayrollCategory
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


class BatchMappingRequest(BaseModel):
    """Batch lookup request for known and unresolved prior codes."""

    model_config = ConfigDict(extra="forbid")

    mode: PrecedenceMode
    priorCodes: list[str] = Field(min_length=1)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str | PrecedenceMode) -> str | PrecedenceMode:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("priorCodes")
    @classmethod
    def normalize_prior_codes(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            code = str(value).strip().upper()
            if not code:
                raise ValueError("priorCodes cannot contain empty values")
            normalized.append(code)
        return normalized


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
    """Admin response containing known prior codes in source order by EDT bucket."""

    totalPriorCodes: int
    Earnings: list[str] = Field(default_factory=list)
    Deductions: list[str] = Field(default_factory=list)
    Taxes: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard sanitized error envelope."""

    error: str
    detail: str | None = None
    statusCode: int
