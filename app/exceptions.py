"""Domain exception hierarchy for the Payroll Mapping Engine."""

from __future__ import annotations


class PayrollMappingError(Exception):
    """Base class for application-owned failures."""

    public_message = "The payroll mapping service could not complete the request"


class ConfigurationError(PayrollMappingError, ValueError):
    """Raised when runtime configuration is invalid."""

    public_message = "The service configuration is invalid"


class DatasetError(PayrollMappingError):
    """Base class for dataset ingestion and validation failures."""

    public_message = "The mapping dataset is invalid or unavailable"


class DatasetLoadError(DatasetError, RuntimeError):
    """Raised when the dataset cannot be read or decoded."""

    public_message = "The mapping dataset could not be loaded"


class DatasetSchemaError(DatasetError, ValueError):
    """Raised when the dataset shape is invalid."""

    public_message = "The mapping dataset failed server-side validation"


class RecordValidationError(DatasetSchemaError):
    """Raised when one source dataset candidate row is invalid."""


class IndexBuildError(PayrollMappingError, ValueError):
    """Raised when normalized records cannot be indexed."""

    public_message = "The mapping index could not be built"


class MappingError(PayrollMappingError, RuntimeError):
    """Base class for deterministic mapping failures."""

    public_message = "The mapping engine could not resolve the requested mapping"


class UnsupportedPrecedenceModeError(MappingError, ValueError):
    """Raised when a requested precedence mode is not registered."""

    public_message = "The requested precedence mode is not supported"


class UnknownPriorCodeError(MappingError, KeyError):
    """Raised when a single-code lookup references an unknown prior code."""

    public_message = "The requested prior code was not found"


class EngineNotReadyError(MappingError):
    """Raised when API traffic arrives before the in-memory engine is ready."""

    public_message = "The mapping engine is not ready"


class GPTAdjudicationError(PayrollMappingError, RuntimeError):
    """Raised when optional GPT tie adjudication fails validation."""

    public_message = "GPT adjudication failed"


class ValidationError(PayrollMappingError, ValueError):
    """Raised when a business validation rule fails."""

    public_message = "The request or generated mapping failed validation"
