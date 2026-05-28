"""In-memory index builder for payroll mapping records."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TypeVar

from app.exceptions import IndexBuildError
from app.logging_utils import log_extra
from app.schemas import NormalizedRecord

logger = logging.getLogger(__name__)

T = TypeVar("T")

RowsByPrior = Mapping[str, tuple[NormalizedRecord, ...]]
CodesByPrior = Mapping[str, frozenset[str]]
IntMetricsByPrior = Mapping[str, Mapping[str, int]]
DateMetricsByPrior = Mapping[str, Mapping[str, datetime]]


@dataclass(frozen=True)
class MappingIndex:
    """Immutable lookup bundle used by the mapping engine."""

    all_rows: RowsByPrior
    unique_codes: CodesByPrior
    occurrence_counts: IntMetricsByPrior
    latest_dates: DateMetricsByPrior
    first_seen_order: IntMetricsByPrior
    all_internal_codes: frozenset[str]
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
        """Return bounded internal evidence for logging or GPT adjudication."""

        selected = list(candidates or sorted(self.unique_codes[prior_code]))
        counts = self.occurrence_counts[prior_code]
        dates = self.latest_dates[prior_code]
        first_seen = self.first_seen_order[prior_code]
        return [
            {
                "internalCode": code,
                "occurrenceCount": counts[code],
                "latestDate": dates[code].strftime("%Y-%m-%d"),
                "firstSeenOrder": first_seen[code],
            }
            for code in selected
        ]


def build_index(records: Sequence[NormalizedRecord]) -> MappingIndex:
    """Build all lookup structures in one pass over normalized records."""

    if not records:
        raise IndexBuildError("Cannot build index from an empty record list")

    all_rows: dict[str, list[NormalizedRecord]] = defaultdict(list)
    unique_codes: dict[str, set[str]] = defaultdict(set)
    occurrence_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latest_dates: dict[str, dict[str, datetime]] = defaultdict(dict)
    first_seen_order: dict[str, dict[str, int]] = defaultdict(dict)
    internal_code_catalog: set[str] = set()
    prior_order: list[str] = []

    for record in records:
        prior_code = record.priorCode
        internal_code = record.internalCode

        if prior_code not in all_rows:
            prior_order.append(prior_code)

        all_rows[prior_code].append(record)
        unique_codes[prior_code].add(internal_code)
        internal_code_catalog.add(internal_code)
        occurrence_counts[prior_code][internal_code] += 1

        previous_date = latest_dates[prior_code].get(internal_code)
        if previous_date is None or record.lastModifiedDate > previous_date:
            latest_dates[prior_code][internal_code] = record.lastModifiedDate

        if internal_code not in first_seen_order[prior_code]:
            first_seen_order[prior_code][internal_code] = record.candidateIndex

    index = MappingIndex(
        all_rows=_freeze_mapping({key: tuple(value) for key, value in all_rows.items()}),
        unique_codes=_freeze_mapping(
            {key: frozenset(value) for key, value in unique_codes.items()}
        ),
        occurrence_counts=_freeze_nested_ints(occurrence_counts),
        latest_dates=_freeze_nested_dates(latest_dates),
        first_seen_order=_freeze_nested_ints(first_seen_order),
        all_internal_codes=frozenset(internal_code_catalog),
        prior_codes=tuple(prior_order),
        total_records=len(records),
    )

    summary = index.summary()
    logger.info(
        "Built mapping index: %d prior codes, %d records, %d one-to-one, %d ambiguous",
        summary["totalPriorCodes"],
        summary["totalRecords"],
        summary["oneToOnePriorCodes"],
        summary["ambiguousPriorCodes"],
        extra=log_extra(
            "mapping_index_built",
            prior_code_count=summary["totalPriorCodes"],
            record_count=summary["totalRecords"],
            one_to_one_count=summary["oneToOnePriorCodes"],
            ambiguous_count=summary["ambiguousPriorCodes"],
        ),
    )
    return index


def _freeze_mapping(value: Mapping[str, T]) -> Mapping[str, T]:
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
