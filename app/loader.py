"""Dataset loader and normalizer for nested payroll mapping JSON."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from app.config import DatasetSource, settings
from app.exceptions import DatasetLoadError, DatasetSchemaError, RecordValidationError
from app.logging_utils import log_extra
from app.schemas import NormalizedRecord, RawCandidateRecord

logger = logging.getLogger(__name__)

RawDataset = dict[object, object]


def load_dataset(
    source: str | None = None,
    *,
    path: str | Path | None = None,
    strict: bool = True,
) -> list[NormalizedRecord]:
    """Load the full dataset and return normalized flat records."""

    effective_source = _resolve_source(source)
    raw_json = _read_raw_json(effective_source, path=path)
    records = _normalize(raw_json, strict=strict)
    prior_code_count = len({record.priorCode for record in records})
    logger.info(
        "Loaded payroll dataset from %s: %d prior codes, %d candidate records",
        effective_source.value,
        prior_code_count,
        len(records),
        extra=log_extra(
            "dataset_loaded",
            source=effective_source.value,
            prior_code_count=prior_code_count,
            record_count=len(records),
        ),
    )
    return records


def _resolve_source(source: str | DatasetSource | None) -> DatasetSource:
    if source is None:
        return settings.effective_dataset_source
    if isinstance(source, DatasetSource):
        return source
    normalized = source.strip().lower()
    try:
        return DatasetSource(normalized)
    except ValueError as exc:
        raise DatasetLoadError("source must be 'local', 'azure', or omitted") from exc


def _read_raw_json(source: DatasetSource, *, path: str | Path | None = None) -> RawDataset:
    if source == DatasetSource.AZURE:
        return _read_from_azure()
    return _read_from_local(path=path)


def _read_from_local(*, path: str | Path | None = None) -> RawDataset:
    dataset_path = Path(path) if path is not None else settings.dataset_local_path
    dataset_path = _resolve_local_path(dataset_path)

    if not dataset_path.exists():
        raise DatasetLoadError(f"Dataset file not found: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetLoadError(f"Dataset path is not a file: {dataset_path}")

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f"Invalid JSON in dataset file '{dataset_path}': {exc}") from exc
    except OSError as exc:
        raise DatasetLoadError(f"Cannot read dataset file '{dataset_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetSchemaError(
            f"Top-level dataset must be a JSON object, got {type(raw).__name__}"
        )
    return raw


def _resolve_local_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    project_root = Path(__file__).resolve().parent.parent
    return project_root / path


def _read_from_azure() -> RawDataset:
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except ImportError as exc:
        raise DatasetLoadError("azure-storage-blob is not installed") from exc

    if not settings.azure_storage_connection_string:
        raise DatasetLoadError("Azure source requested but no storage connection is configured")

    try:
        client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        blob_client = client.get_blob_client(
            container=settings.azure_storage_container_name,
            blob=settings.azure_storage_blob_name,
        )
        payload = blob_client.download_blob().readall()
        raw: object = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(
            f"Invalid JSON in Azure blob '{settings.azure_storage_blob_name}': {exc}"
        ) from exc
    except Exception as exc:
        raise DatasetLoadError(f"Failed to read dataset from Azure Blob Storage: {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetSchemaError(
            f"Top-level dataset must be a JSON object, got {type(raw).__name__}"
        )
    logger.info(
        "Azure dataset load completed: container=%s blob=%s",
        settings.azure_storage_container_name,
        settings.azure_storage_blob_name,
        extra=log_extra(
            "azure_dataset_load_completed",
            container=settings.azure_storage_container_name,
            blob=settings.azure_storage_blob_name,
        ),
    )
    return raw


def _normalize(raw: object, *, strict: bool = True) -> list[NormalizedRecord]:
    """Normalize nested JSON into a flat list of validated records."""

    dataset = _validate_top_level(raw)
    records: list[NormalizedRecord] = []
    errors: list[str] = []
    global_index = 0

    for prior_code_raw, candidates in dataset.items():
        try:
            prior_code = _normalize_code(prior_code_raw, field_name="priorCode")
        except RecordValidationError as exc:
            errors.append(str(exc))
            if strict:
                continue
            logger.warning("Skipping invalid prior code key: %s", exc)
            continue

        if not isinstance(candidates, list):
            raise DatasetSchemaError(
                f"Expected a list of candidates for prior code '{prior_code}', "
                f"got {type(candidates).__name__}"
            )
        if not candidates:
            errors.append(f"priorCode '{prior_code}' has an empty candidate list")
            continue

        for candidate_index, candidate in enumerate(candidates):
            try:
                record = _parse_candidate(
                    prior_code=prior_code,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    global_index=global_index,
                )
            except RecordValidationError as exc:
                message = (
                    f"priorCode='{prior_code}' candidateIndex={candidate_index}: {exc}"
                )
                if strict:
                    errors.append(message)
                else:
                    logger.warning("Skipping malformed record: %s", message)
                continue

            records.append(record)
            global_index += 1

    if errors and strict:
        preview = "; ".join(errors[:10])
        if len(errors) > 10:
            preview += f"; ... {len(errors) - 10} more"
        logger.warning(
            "Dataset validation failed with %d error(s)",
            len(errors),
            extra=log_extra("dataset_validation_failed", error_count=len(errors)),
        )
        raise DatasetSchemaError(f"Dataset validation failed: {preview}")

    if not records:
        raise DatasetSchemaError("Dataset contains zero valid mapping records")

    return records


def _validate_top_level(raw: object) -> RawDataset:
    if not isinstance(raw, dict):
        raise DatasetSchemaError(
            f"Top-level dataset must be a JSON object, got {type(raw).__name__}"
        )
    if not raw:
        raise DatasetSchemaError("Dataset is empty: no prior codes found")
    return raw


def _parse_candidate(
    prior_code: str,
    candidate_index: int,
    candidate: object,
    global_index: int = 0,
) -> NormalizedRecord:
    if not isinstance(candidate, dict):
        raise RecordValidationError(
            f"candidate must be an object, got {type(candidate).__name__}"
        )

    try:
        raw_record = RawCandidateRecord.model_validate(candidate)
    except PydanticValidationError as exc:
        raise RecordValidationError(str(exc)) from exc

    internal_code = _normalize_code(raw_record.internalCode, field_name="internalCode")
    parsed_date = _parse_date(raw_record.LastModifiedDate, prior_code, candidate_index)

    return NormalizedRecord(
        priorCode=prior_code,
        internalCode=internal_code,
        lastModifiedDate=parsed_date,
        candidateIndex=candidate_index,
        globalIndex=global_index,
    )


def _normalize_code(value: object, *, field_name: str = "code") -> str:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a string")
    normalized = value.strip().upper()
    if not normalized:
        raise RecordValidationError(f"{field_name} is empty after normalization")
    return normalized


def _parse_date(date_value: object, prior_code: str, candidate_index: int) -> datetime:
    if not isinstance(date_value, str):
        raise RecordValidationError("LastModifiedDate must be a string")
    try:
        return datetime.strptime(date_value.strip(), settings.date_format)
    except ValueError as exc:
        raise RecordValidationError(
            f"Invalid LastModifiedDate '{date_value}' for priorCode='{prior_code}' "
            f"candidateIndex={candidate_index}; expected MM/DD/YYYY"
        ) from exc
