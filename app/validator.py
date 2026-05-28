"""Business validation guards for loaded data and final mappings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import logging

from app.config import PrecedenceMode
from app.exceptions import ValidationError
from app.logging_utils import log_extra
from app.schemas import MappingResult, NormalizedRecord

logger = logging.getLogger(__name__)


def validate_mapping_request_mode(mode: str | PrecedenceMode) -> PrecedenceMode:
    if isinstance(mode, PrecedenceMode):
        return mode
    try:
        return PrecedenceMode(str(mode).strip().upper())
    except ValueError as exc:
        raise ValidationError(
            f"Invalid precedence mode '{mode}'. Accepted values: "
            f"{[item.value for item in PrecedenceMode]}"
        ) from exc


def validate_loaded_records(records: Sequence[NormalizedRecord]) -> None:
    if not records:
        logger.error(
            "Dataset validation failed: no records",
            extra=log_extra("dataset_validation_failed"),
        )
        raise ValidationError("Dataset is empty")

    prior_codes = {record.priorCode for record in records}
    internal_codes = {record.internalCode for record in records}
    if not prior_codes:
        raise ValidationError("Dataset contains no prior codes")
    if not internal_codes:
        raise ValidationError("Dataset contains no internal codes")

    bad_dates = [record for record in records if not isinstance(record.lastModifiedDate, datetime)]
    if bad_dates:
        raise ValidationError(f"{len(bad_dates)} records have invalid parsed dates")

    empty_codes = [
        record for record in records if not record.priorCode.strip() or not record.internalCode.strip()
    ]
    if empty_codes:
        raise ValidationError(f"{len(empty_codes)} records contain empty codes")

    logger.info(
        "Dataset validation passed: %d prior codes, %d internal codes, %d records",
        len(prior_codes),
        len(internal_codes),
        len(records),
        extra=log_extra(
            "dataset_validation_passed",
            prior_code_count=len(prior_codes),
            internal_code_count=len(internal_codes),
            record_count=len(records),
        ),
    )


def validate_mapping_results(
    results: Sequence[MappingResult],
    index_prior_codes: Sequence[str],
) -> None:
    if not results:
        raise ValidationError("Mapping produced zero results")

    expected = list(index_prior_codes)
    actual = [result.priorCode for result in results]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValidationError(
            "Mapping result prior-code set/order mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if len(actual) != len(set(actual)):
        duplicates = sorted({code for code in actual if actual.count(code) > 1})
        raise ValidationError(f"Duplicate prior codes in mapping results: {duplicates}")

    empty_internal = [result.priorCode for result in results if not result.internalCode.strip()]
    if empty_internal:
        raise ValidationError(
            f"{len(empty_internal)} mappings have empty internalCode: {empty_internal}"
        )

    for result in results:
        if set(result.model_dump().keys()) != {"priorCode", "internalCode"}:
            raise ValidationError(f"Mapping result leaked internal fields: {result}")
