"""Business validation guards for loaded data and final mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import logging

from app.config import PAYROLL_CATEGORIES, PayrollCategory, PrecedenceMode
from app.exceptions import ValidationError
from app.logging_utils import log_extra
from app.prompt_builder import NO_MATCH_GLOBAL_CODE
from app.schemas import CategoryMappingResponse, NormalizedRecord

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

    category_scoped_prior_codes = {(record.category, record.priorCode) for record in records}
    global_codes = {record.globalCode for record in records}
    categories = {record.category for record in records}
    if not category_scoped_prior_codes:
        raise ValidationError("Dataset contains no category-scoped prior codes")
    if not global_codes:
        raise ValidationError("Dataset contains no global codes")
    if not categories <= set(PAYROLL_CATEGORIES):
        raise ValidationError("Dataset contains unsupported categories")

    bad_dates = [record for record in records if not isinstance(record.lastModifiedDate, datetime)]
    if bad_dates:
        raise ValidationError(f"{len(bad_dates)} records have invalid parsed dates")

    empty_codes = [
        record for record in records if not record.priorCode.strip() or not record.globalCode.strip()
    ]
    if empty_codes:
        raise ValidationError(f"{len(empty_codes)} records contain empty codes")

    logger.info(
        "Dataset validation passed: %d categories, %d category-scoped prior codes, "
        "%d global codes, %d records",
        len(categories),
        len(category_scoped_prior_codes),
        len(global_codes),
        len(records),
        extra=log_extra(
            "dataset_validation_passed",
            category_count=len(categories),
            prior_code_count=len(category_scoped_prior_codes),
            global_code_count=len(global_codes),
            record_count=len(records),
        ),
    )


def validate_mapping_results(
    results: CategoryMappingResponse,
    expected_prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
) -> None:
    """Validate deterministic map-all output order and public contract."""

    for category in PAYROLL_CATEGORIES:
        category_results = results.as_category_map()[category]
        expected = list(expected_prior_codes_by_category.get(category, ()))
        actual = [result.priorCode for result in category_results]
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValidationError(
                f"Mapping result prior-code set/order mismatch for {category.value}. "
                f"missing={missing}, extra={extra}"
            )
        _validate_mapping_items(category_results)


def validate_category_response_contract(results: CategoryMappingResponse) -> None:
    for category_results in results.as_category_map().values():
        _validate_mapping_items(category_results)


def validate_gpt_mapping_response(
    results: CategoryMappingResponse,
    allowed_catalogs: Mapping[PayrollCategory, Sequence[str]],
) -> None:
    allowed = {
        category: {str(code).strip().upper() for code in allowed_catalogs.get(category, ())}
        for category in PAYROLL_CATEGORIES
    }
    for category, category_results in results.as_category_map().items():
        _validate_mapping_items(category_results)
        for result in category_results:
            if result.globalCode == NO_MATCH_GLOBAL_CODE:
                continue
            if result.globalCode not in allowed[category]:
                raise ValidationError(
                    f"GPT returned globalCode '{result.globalCode}' outside {category.value} catalog"
                )


def _validate_mapping_items(category_results: Sequence[object]) -> None:
    for result in category_results:
        if not hasattr(result, "model_dump"):
            raise ValidationError(f"Mapping result is not a Pydantic model: {result}")
        payload = result.model_dump()
        if set(payload.keys()) != {"priorCode", "globalCode"}:
            raise ValidationError(f"Mapping result leaked non-contract fields: {result}")
        if not str(payload["priorCode"]).strip():
            raise ValidationError("Mapping result contains an empty priorCode")
        if not str(payload["globalCode"]).strip():
            raise ValidationError(f"Mapping result has empty globalCode: {result}")
