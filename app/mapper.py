"""Deterministic category-scoped payroll mapping adjudication engine."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias

from app.config import PAYROLL_CATEGORIES, PayrollCategory, PrecedenceMode, TieBreakStrategy, settings
from app.exceptions import (
    GPTAdjudicationError,
    MappingError,
    UnsupportedPrecedenceModeError,
)
from app.index_builder import CategoryMappingIndex, MappingIndex
from app.logging_utils import log_extra
from app.prompt_builder import NO_MATCH_GLOBAL_CODE
from app.schemas import CategoryMappingResponse, MappingDecisionDetail, MappingResult

logger = logging.getLogger(__name__)

MetricValue: TypeAlias = int | datetime
MetricValues: TypeAlias = Mapping[str, MetricValue]
RankingRule: TypeAlias = tuple[str, MetricValues]
CategoryResultMap: TypeAlias = dict[PayrollCategory, list[MappingResult]]


class GPTAdjudicator(Protocol):
    """Minimal interface required from optional GPT fallback providers."""

    def recommend_global_code(
        self,
        *,
        prior_code: str,
        catalogs: Mapping[PayrollCategory, Sequence[str]],
        catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, object]]],
    ) -> CategoryMappingResponse:
        """Recommend category-scoped global-code mappings for one missing prior code."""

    def recommend_global_codes(
        self,
        *,
        prior_codes: Sequence[str],
        catalogs: Mapping[PayrollCategory, Sequence[str]],
        catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, object]]],
    ) -> CategoryMappingResponse:
        """Recommend category-scoped global-code mappings for missing prior codes."""


@dataclass(frozen=True)
class MappingResolution:
    """Internal deterministic decision before public response shaping."""

    winner: str
    detail: MappingDecisionDetail


ModeResolver: TypeAlias = Callable[[CategoryMappingIndex, str], MappingResolution]
ResolverRegistrar: TypeAlias = Callable[[ModeResolver], ModeResolver]

MODE_RESOLVERS: dict[PrecedenceMode, ModeResolver] = {}


def register_mode_resolver(mode: PrecedenceMode) -> ResolverRegistrar:
    """Register a resolver for a precedence mode.

    New deterministic modes should define one resolver and register it here.
    Core orchestration uses the registry only and contains no mode branching.
    """

    def decorator(resolver: ModeResolver) -> ModeResolver:
        MODE_RESOLVERS[mode] = resolver
        return resolver

    return decorator


def supported_modes() -> tuple[PrecedenceMode, ...]:
    """Return the precedence modes currently registered with the engine."""

    return tuple(MODE_RESOLVERS)


def map_all(
    index: MappingIndex,
    mode: PrecedenceMode | str,
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve every historical prior code within each category namespace."""

    del gpt_client
    resolved_mode = _coerce_mode(mode)
    logger.info(
        "Category-scoped batch mapping started: mode=%s priorCodes=%d",
        resolved_mode.value,
        index.total_prior_codes,
        extra=log_extra(
            "mapping_started",
            mode=resolved_mode.value,
            prior_code_count=index.total_prior_codes,
        ),
    )

    results = _empty_result_map()
    for category in PAYROLL_CATEGORIES:
        category_index = index.category_indexes[category]
        results[category] = [
            _resolve(index=category_index, prior_code=prior_code, mode=resolved_mode)
            for prior_code in category_index.prior_codes
        ]

    total_mapped = sum(len(items) for items in results.values())
    logger.info(
        "Category-scoped batch mapping complete: mode=%s mapped=%d",
        resolved_mode.value,
        total_mapped,
        extra=log_extra("mapping_completed", mode=resolved_mode.value, mapped_count=total_mapped),
    )
    return CategoryMappingResponse.from_category_map(results)


def map_one(
    index: MappingIndex,
    prior_code: str,
    mode: PrecedenceMode | str,
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve one prior code across category namespaces."""

    normalized_prior_code = _normalize_prior_code(prior_code)
    resolved_mode = _coerce_mode(mode)
    matching_categories = index.categories_for_prior_code(normalized_prior_code)
    if not matching_categories:
        return _resolve_missing_prior_with_gpt(
            prior_code=normalized_prior_code,
            index=index,
            gpt_client=gpt_client,
        )

    results = _empty_result_map()
    for category in matching_categories:
        results[category].append(
            _resolve(
                index=index.category_indexes[category],
                prior_code=normalized_prior_code,
                mode=resolved_mode,
            )
        )
    return CategoryMappingResponse.from_category_map(results)


def map_batch(
    index: MappingIndex,
    prior_codes: Sequence[str],
    mode: PrecedenceMode | str,
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve known prior codes deterministically and missing ones through GPT fallback."""

    normalized_prior_codes = [_normalize_prior_code(prior_code) for prior_code in prior_codes]
    resolved_mode = _coerce_mode(mode)
    results = _empty_result_map()
    missing_prior_codes: list[str] = []

    for prior_code in normalized_prior_codes:
        matching_categories = index.categories_for_prior_code(prior_code)
        if not matching_categories:
            missing_prior_codes.append(prior_code)
            continue

        for category in matching_categories:
            results[category].append(
                _resolve(
                    index=index.category_indexes[category],
                    prior_code=prior_code,
                    mode=resolved_mode,
                )
            )

    if missing_prior_codes:
        fallback = _resolve_missing_priors_with_gpt(
            prior_codes=missing_prior_codes,
            index=index,
            gpt_client=gpt_client,
        )
        _merge_response(results, fallback)

    return CategoryMappingResponse.from_category_map(results)


def _resolve(
    *,
    index: CategoryMappingIndex,
    prior_code: str,
    mode: PrecedenceMode,
) -> MappingResult:
    resolver = _resolver_for_mode(mode)
    resolution = resolver(index, prior_code)
    detail = resolution.detail

    if detail.tieBreakApplied:
        logger.info(
            "Tie-break triggered: category=%s priorCode=%s mode=%s strategy=%s",
            detail.category.value,
            detail.priorCode,
            detail.mode.value,
            detail.tieBreakStrategy,
            extra=log_extra(
                "tie_break_triggered",
                category=detail.category.value,
                prior_code=detail.priorCode,
                mode=detail.mode.value,
                strategy=detail.tieBreakStrategy,
            ),
        )

    _log_decision(detail)
    return MappingResult(priorCode=prior_code, globalCode=resolution.winner)


def _resolve_missing_prior_with_gpt(
    *,
    prior_code: str,
    index: MappingIndex,
    gpt_client: GPTAdjudicator | None,
) -> CategoryMappingResponse:
    """Resolve one missing prior code using category-wise GPT fallback."""

    return _resolve_missing_priors_with_gpt(
        prior_codes=[prior_code],
        index=index,
        gpt_client=gpt_client,
    )


def _resolve_missing_priors_with_gpt(
    *,
    prior_codes: Sequence[str],
    index: MappingIndex,
    gpt_client: GPTAdjudicator | None,
) -> CategoryMappingResponse:
    """Resolve missing prior codes using GPT against category-specific catalogs only."""

    catalogs = index.all_global_codes_by_category()
    total_candidates = sum(len(codes) for codes in catalogs.values())
    if total_candidates == 0:
        logger.warning(
            "Missing prior-code fallback has no global-code candidates",
            extra=log_extra("missing_prior_no_candidates", prior_code_count=len(prior_codes)),
        )
        return _no_match_response(prior_codes)

    logger.info(
        "Missing prior-code GPT fallback eligible: priorCodes=%d candidates=%d",
        len(prior_codes),
        total_candidates,
        extra=log_extra(
            "missing_prior_gpt_fallback_eligible",
            prior_code_count=len(prior_codes),
            candidate_count=total_candidates,
        ),
    )

    if gpt_client is None:
        logger.warning(
            "Missing prior-code GPT fallback unavailable; returning NO_MATCH",
            extra=log_extra("missing_prior_gpt_unavailable", prior_code_count=len(prior_codes)),
        )
        return _no_match_response(prior_codes)

    try:
        if len(prior_codes) == 1:
            return gpt_client.recommend_global_code(
                prior_code=prior_codes[0],
                catalogs=catalogs,
                catalog_evidence=index.catalog_evidence_by_category(),
            )
        return gpt_client.recommend_global_codes(
            prior_codes=prior_codes,
            catalogs=catalogs,
            catalog_evidence=index.catalog_evidence_by_category(),
        )
    except GPTAdjudicationError as exc:
        logger.warning(
            "Missing prior-code GPT fallback failed: %s",
            exc,
            extra=log_extra("missing_prior_gpt_failed", prior_code_count=len(prior_codes)),
        )
        return _no_match_response(prior_codes)
    except Exception:
        logger.exception(
            "Unexpected missing prior-code GPT fallback failure",
            extra=log_extra(
                "missing_prior_gpt_unexpected_failure",
                prior_code_count=len(prior_codes),
            ),
        )
        return _no_match_response(prior_codes)


@register_mode_resolver(PrecedenceMode.ONE_TO_ONE)
def _resolve_one_to_one(index: CategoryMappingIndex, prior_code: str) -> MappingResolution:
    unique_codes = sorted(index.unique_codes[prior_code])

    if len(unique_codes) == 1:
        winner = unique_codes[0]
        tied_candidates = [winner]
        tie_break_applied = False
        tie_strategy = None
        secondary_rule = None
    else:
        winner, tied_candidates, tie_break_applied, tie_strategy = _rank_candidates(
            index=index,
            prior_code=prior_code,
            candidates=unique_codes,
            ranking=[
                ("occurrence_count", _count_metric(index, prior_code)),
                ("latest_date", _date_metric(index, prior_code)),
            ],
        )
        secondary_rule = "not_one_to_one_fallback: occurrence_count -> latest_date"

    return MappingResolution(
        winner=winner,
        detail=_detail(
            index=index,
            prior_code=prior_code,
            mode=PrecedenceMode.ONE_TO_ONE,
            winner=winner,
            primary_rule="single_unique_global_code",
            secondary_rule=secondary_rule,
            tied_candidates=tied_candidates,
            tie_break_applied=tie_break_applied,
            tie_strategy=tie_strategy,
        ),
    )


@register_mode_resolver(PrecedenceMode.MAX_OCCURRENCE)
def _resolve_max_occurrence(index: CategoryMappingIndex, prior_code: str) -> MappingResolution:
    winner, tied_candidates, tie_break_applied, tie_strategy = _rank_candidates(
        index=index,
        prior_code=prior_code,
        candidates=sorted(index.unique_codes[prior_code]),
        ranking=[
            ("occurrence_count", _count_metric(index, prior_code)),
            ("latest_date", _date_metric(index, prior_code)),
        ],
    )
    return MappingResolution(
        winner=winner,
        detail=_detail(
            index=index,
            prior_code=prior_code,
            mode=PrecedenceMode.MAX_OCCURRENCE,
            winner=winner,
            primary_rule="highest_occurrence_count",
            secondary_rule="latest_date_when_count_ties",
            tied_candidates=tied_candidates,
            tie_break_applied=tie_break_applied,
            tie_strategy=tie_strategy,
        ),
    )


@register_mode_resolver(PrecedenceMode.LAST_MODIFIED_DATE)
def _resolve_last_modified_date(index: CategoryMappingIndex, prior_code: str) -> MappingResolution:
    winner, tied_candidates, tie_break_applied, tie_strategy = _rank_candidates(
        index=index,
        prior_code=prior_code,
        candidates=sorted(index.unique_codes[prior_code]),
        ranking=[
            ("latest_date", _date_metric(index, prior_code)),
            ("occurrence_count", _count_metric(index, prior_code)),
        ],
    )
    return MappingResolution(
        winner=winner,
        detail=_detail(
            index=index,
            prior_code=prior_code,
            mode=PrecedenceMode.LAST_MODIFIED_DATE,
            winner=winner,
            primary_rule="most_recent_last_modified_date",
            secondary_rule="occurrence_count_when_date_ties",
            tied_candidates=tied_candidates,
            tie_break_applied=tie_break_applied,
            tie_strategy=tie_strategy,
        ),
    )


def _rank_candidates(
    *,
    index: CategoryMappingIndex,
    prior_code: str,
    candidates: Sequence[str],
    ranking: Sequence[RankingRule],
) -> tuple[str, list[str], bool, str | None]:
    """Apply primary and secondary metrics, then hidden deterministic tie-break."""

    remaining = list(candidates)
    if not remaining:
        raise MappingError(f"No candidates available for prior code '{prior_code}'")

    for _rule_name, metric_values in ranking:
        best_value = max(metric_values[code] for code in remaining)
        remaining = [code for code in remaining if metric_values[code] == best_value]
        if len(remaining) == 1:
            return remaining[0], remaining, False, None

    winner = _break_tie(index=index, prior_code=prior_code, candidates=remaining)
    return winner, sorted(remaining), True, _tie_break_strategy().value


def _break_tie(index: CategoryMappingIndex, prior_code: str, candidates: Sequence[str]) -> str:
    strategy = _tie_break_strategy()
    if strategy == TieBreakStrategy.LEXICOGRAPHIC:
        return sorted(candidates)[0]

    first_seen = index.first_seen_order[prior_code]
    return sorted(candidates, key=lambda code: (first_seen.get(code, 10**9), code))[0]


def _count_metric(index: CategoryMappingIndex, prior_code: str) -> Mapping[str, int]:
    return index.occurrence_counts[prior_code]


def _date_metric(index: CategoryMappingIndex, prior_code: str) -> Mapping[str, datetime]:
    return index.latest_dates[prior_code]


def _detail(
    *,
    index: CategoryMappingIndex,
    prior_code: str,
    mode: PrecedenceMode,
    winner: str,
    primary_rule: str,
    secondary_rule: str | None,
    tied_candidates: Sequence[str],
    tie_break_applied: bool,
    tie_strategy: str | None,
) -> MappingDecisionDetail:
    latest_dates = {
        code: date.strftime("%Y-%m-%d")
        for code, date in index.latest_dates[prior_code].items()
    }
    return MappingDecisionDetail(
        category=index.category,
        priorCode=prior_code,
        winningCode=winner,
        mode=mode,
        primaryRule=primary_rule,
        secondaryRule=secondary_rule,
        candidateCount=index.candidate_count(prior_code),
        uniqueCandidates=sorted(index.unique_codes[prior_code]),
        tiedCandidates=sorted(tied_candidates),
        occurrenceCounts=dict(index.occurrence_counts[prior_code]),
        latestDates=latest_dates,
        tieBreakApplied=tie_break_applied,
        tieBreakStrategy=tie_strategy,
    )


def _resolver_for_mode(mode: PrecedenceMode) -> ModeResolver:
    resolver = MODE_RESOLVERS.get(mode)
    if resolver is None:
        supported = [registered_mode.value for registered_mode in supported_modes()]
        raise UnsupportedPrecedenceModeError(
            f"Unsupported precedence mode '{mode.value}'. Registered modes: {supported}"
        )
    return resolver


def _coerce_mode(mode: PrecedenceMode | str) -> PrecedenceMode:
    if isinstance(mode, PrecedenceMode):
        return mode
    try:
        return PrecedenceMode(str(mode).strip().upper())
    except ValueError as exc:
        supported = [registered_mode.value for registered_mode in supported_modes()]
        raise UnsupportedPrecedenceModeError(
            f"Unsupported precedence mode '{mode}'. Registered modes: {supported}"
        ) from exc


def _tie_break_strategy() -> TieBreakStrategy:
    value = settings.tie_break_strategy
    if isinstance(value, TieBreakStrategy):
        return value
    try:
        return TieBreakStrategy(str(value).strip().lower())
    except ValueError:
        logger.warning(
            "Invalid tie break strategy configured; defaulting to first_seen",
            extra=log_extra("invalid_tie_break_strategy"),
        )
        return TieBreakStrategy.FIRST_SEEN


def _normalize_prior_code(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise MappingError("priorCode cannot be empty")
    return normalized


def _empty_result_map() -> CategoryResultMap:
    return {category: [] for category in PAYROLL_CATEGORIES}


def _merge_response(target: CategoryResultMap, response: CategoryMappingResponse) -> None:
    for category, items in response.as_category_map().items():
        target[category].extend(items)


def _no_match_response(prior_codes: Sequence[str]) -> CategoryMappingResponse:
    return CategoryMappingResponse.from_category_map(
        {
            category: [
                MappingResult(priorCode=prior_code, globalCode=NO_MATCH_GLOBAL_CODE)
                for prior_code in prior_codes
            ]
            for category in PAYROLL_CATEGORIES
        }
    )


def _log_decision(detail: MappingDecisionDetail) -> None:
    logger.debug(
        "MAP_DECISION category=%s priorCode=%s mode=%s winner=%s candidates=%d unique=%d "
        "primary=%s secondary=%s tie=%s gpt=%s",
        detail.category.value,
        detail.priorCode,
        detail.mode.value,
        detail.winningCode,
        detail.candidateCount,
        len(detail.uniqueCandidates),
        detail.primaryRule,
        detail.secondaryRule,
        detail.tieBreakApplied,
        detail.gptAdjudicated,
        extra=log_extra(
            "mapping_decision",
            category=detail.category.value,
            prior_code=detail.priorCode,
            mode=detail.mode.value,
            winner=detail.winningCode,
            candidate_count=detail.candidateCount,
            unique_candidate_count=len(detail.uniqueCandidates),
            tie_break_applied=detail.tieBreakApplied,
            gpt_adjudicated=detail.gptAdjudicated,
        ),
    )
