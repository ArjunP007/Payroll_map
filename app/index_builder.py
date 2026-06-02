"""In-memory category-scoped indexes for payroll mapping records."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, TypeVar

from app.config import PAYROLL_CATEGORIES, PayrollCategory
from app.exceptions import IndexBuildError
from app.logging_utils import log_extra
from app.schemas import NormalizedRecord

logger = logging.getLogger(__name__)

T = TypeVar("T")

RowsByPrior = Mapping[str, tuple[NormalizedRecord, ...]]
CodesByPrior = Mapping[str, frozenset[str]]
IntMetricsByPrior = Mapping[str, Mapping[str, int]]
DateMetricsByPrior = Mapping[str, Mapping[str, datetime]]
CategoryIndexes = Mapping[PayrollCategory, "CategoryMappingIndex"]


@dataclass(frozen=True)
class CategoryMappingIndex:
    """Immutable lookup bundle for one payroll category namespace."""

    category: PayrollCategory
    all_rows: RowsByPrior
    unique_codes: CodesByPrior
    occurrence_counts: IntMetricsByPrior
    latest_dates: DateMetricsByPrior
    first_seen_order: IntMetricsByPrior
    all_global_codes: frozenset[str]
    prior_codes: tuple[str, ...]
    total_records: int

    def is_one_to_one(self, prior_code: str) -> bool:
        return len(self.unique_codes.get(prior_code, frozenset())) == 1

    def candidate_count(self, prior_code: str) -> int:
        return len(self.all_rows.get(prior_code, ()))

    def unique_code_count(self, prior_code: str) -> int:
        return len(self.unique_codes.get(prior_code, frozenset()))

    def summary(self) -> dict[str, int]:
        one_to_one = sum(1 for prior_code in self.prior_codes if self.is_one_to_one(prior_code))
        return {
            "totalPriorCodes": len(self.prior_codes),
            "totalRecords": self.total_records,
            "oneToOnePriorCodes": one_to_one,
            "ambiguousPriorCodes": len(self.prior_codes) - one_to_one,
        }

    def candidate_evidence(
        self,
        prior_code: str,
        candidates: Sequence[str] | None = None,
    ) -> list[dict[str, str | int]]:
        """Return bounded deterministic evidence for logging or diagnostics."""

        selected = list(candidates or sorted(self.unique_codes[prior_code]))
        counts = self.occurrence_counts[prior_code]
        dates = self.latest_dates[prior_code]
        first_seen = self.first_seen_order[prior_code]
        return [
            {
                "globalCode": code,
                "occurrenceCount": counts[code],
                "latestDate": dates[code].strftime("%Y-%m-%d"),
                "firstSeenOrder": first_seen[code],
            }
            for code in selected
        ]

    def catalog_evidence(self) -> list[dict[str, Any]]:
        """Return metadata-rich candidate rows for GPT fallback prompts."""

        evidence: list[dict[str, Any]] = []
        for prior_code in self.prior_codes:
            for record in self.all_rows[prior_code]:
                item: dict[str, Any] = {
                    "priorCode": record.priorCode,
                    "globalCode": record.globalCode,
                    "LastModifiedDate": record.lastModifiedDate.strftime("%Y-%m-%d"),
                }
                if record.metadata:
                    item["metadata"] = dict(record.metadata)
                evidence.append(item)
        return evidence


@dataclass(frozen=True)
class MappingIndex:
    """Immutable category-scoped lookup bundle used by the mapping engine."""

    category_indexes: CategoryIndexes
    categories: tuple[PayrollCategory, ...]
    total_records: int

    @property
    def total_prior_codes(self) -> int:
        return sum(len(index.prior_codes) for index in self.category_indexes.values())

    def prior_codes_by_category(self) -> dict[PayrollCategory, tuple[str, ...]]:
        return {
            category: self.category_indexes[category].prior_codes
            for category in self.categories
        }

    def all_global_codes_by_category(self) -> dict[PayrollCategory, tuple[str, ...]]:
        return {
            category: tuple(sorted(self.category_indexes[category].all_global_codes))
            for category in self.categories
        }

    def catalog_evidence_by_category(self) -> dict[PayrollCategory, list[dict[str, Any]]]:
        return {
            category: self.category_indexes[category].catalog_evidence()
            for category in self.categories
        }

    def categories_for_prior_code(self, prior_code: str) -> tuple[PayrollCategory, ...]:
        return tuple(
            category
            for category in self.categories
            if prior_code in self.category_indexes[category].all_rows
        )

    def summary(self) -> dict[str, int]:
        return {
            "totalCategories": len(self.categories),
            "totalPriorCodes": self.total_prior_codes,
            "totalRecords": self.total_records,
        }


def build_index(records: Sequence[NormalizedRecord]) -> MappingIndex:
    """Build category-scoped lookup structures in one pass over normalized records."""

    if not records:
        raise IndexBuildError("Cannot build index from an empty record list")

    by_category: dict[PayrollCategory, list[NormalizedRecord]] = {
        category: [] for category in PAYROLL_CATEGORIES
    }
    for record in records:
        by_category[record.category].append(record)

    category_indexes = {
        category: _build_category_index(category, by_category[category])
        for category in PAYROLL_CATEGORIES
    }
    index = MappingIndex(
        category_indexes=_freeze_mapping(category_indexes),
        categories=PAYROLL_CATEGORIES,
        total_records=len(records),
    )

    summary = index.summary()
    logger.info(
        "Built category-scoped mapping index: %d categories, %d prior codes, %d records",
        summary["totalCategories"],
        summary["totalPriorCodes"],
        summary["totalRecords"],
        extra=log_extra(
            "mapping_index_built",
            category_count=summary["totalCategories"],
            prior_code_count=summary["totalPriorCodes"],
            record_count=summary["totalRecords"],
        ),
    )
    return index


def _build_category_index(
    category: PayrollCategory,
    records: Sequence[NormalizedRecord],
) -> CategoryMappingIndex:
    all_rows: dict[str, list[NormalizedRecord]] = defaultdict(list)
    unique_codes: dict[str, set[str]] = defaultdict(set)
    occurrence_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latest_dates: dict[str, dict[str, datetime]] = defaultdict(dict)
    first_seen_order: dict[str, dict[str, int]] = defaultdict(dict)
    global_code_catalog: set[str] = set()
    prior_order: list[str] = []

    for record in records:
        prior_code = record.priorCode
        global_code = record.globalCode

        if prior_code not in all_rows:
            prior_order.append(prior_code)

        all_rows[prior_code].append(record)
        unique_codes[prior_code].add(global_code)
        global_code_catalog.add(global_code)
        occurrence_counts[prior_code][global_code] += 1

        previous_date = latest_dates[prior_code].get(global_code)
        if previous_date is None or record.lastModifiedDate > previous_date:
            latest_dates[prior_code][global_code] = record.lastModifiedDate

        if global_code not in first_seen_order[prior_code]:
            first_seen_order[prior_code][global_code] = record.candidateIndex

    return CategoryMappingIndex(
        category=category,
        all_rows=_freeze_mapping({key: tuple(value) for key, value in all_rows.items()}),
        unique_codes=_freeze_mapping(
            {key: frozenset(value) for key, value in unique_codes.items()}
        ),
        occurrence_counts=_freeze_nested_ints(occurrence_counts),
        latest_dates=_freeze_nested_dates(latest_dates),
        first_seen_order=_freeze_nested_ints(first_seen_order),
        all_global_codes=frozenset(global_code_catalog),
        prior_codes=tuple(prior_order),
        total_records=len(records),
    )


def _freeze_mapping(value: Mapping[T, Any]) -> Mapping[T, Any]:
    return MappingProxyType(dict(value))


def _freeze_nested_ints(value: Mapping[str, Mapping[str, int]]) -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            outer_key: MappingProxyType(dict(inner_value))
            for outer_key, inner_value in value.items()
        }
    )


def _freeze_nested_dates(
    value: Mapping[str, Mapping[str, datetime]]
) -> Mapping[str, Mapping[str, datetime]]:
    return MappingProxyType(
        {
            outer_key: MappingProxyType(dict(inner_value))
            for outer_key, inner_value in value.items()
        }
    )
