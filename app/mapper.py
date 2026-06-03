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
        category: PayrollCategory,
        catalogs: Mapping[PayrollCategory, Sequence[str]],
        catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, object]]],
    ) -> CategoryMappingResponse:
        """Recommend category-scoped global-code mappings for one missing prior code."""

    def recommend_global_codes(
        self,
        *,
        prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
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
    categories: Sequence[PayrollCategory | str],
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve every historical prior code within each category namespace."""

    del gpt_client
    resolved_mode = _coerce_mode(mode)
    selected_categories = _normalize_categories(categories)
    selected_prior_count = sum(
        len(index.category_indexes[category].prior_codes)
        for category in selected_categories
    )
    logger.info(
        "Category-scoped batch mapping started: mode=%s priorCodes=%d",
        resolved_mode.value,
        selected_prior_count,
        extra=log_extra(
            "mapping_started",
            mode=resolved_mode.value,
            prior_code_count=selected_prior_count,
            categories=[category.value for category in selected_categories],
        ),
    )

    results = _empty_result_map()
    for category in selected_categories:
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
    categories: Sequence[PayrollCategory | str],
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve one prior code across category namespaces."""

    normalized_prior_code = _normalize_prior_code(prior_code)
    resolved_mode = _coerce_mode(mode)
    selected_categories = _normalize_categories(categories)
    results = _empty_result_map()
    missing_by_category = _empty_missing_map(selected_categories)

    for category in selected_categories:
        category_index = index.category_indexes[category]
        if normalized_prior_code in category_index.all_rows:
            results[category].append(
                _resolve(
                    index=category_index,
                    prior_code=normalized_prior_code,
                    mode=resolved_mode,
                )
            )
        else:
            missing_by_category[category].append(normalized_prior_code)

    if _has_missing(missing_by_category):
        fallback = _resolve_missing_priors_with_gpt(
            missing_prior_codes_by_category=missing_by_category,
            index=index,
            gpt_client=gpt_client,
        )
        _merge_response(results, fallback)
    return CategoryMappingResponse.from_category_map(results)


def map_batch(
    index: MappingIndex,
    prior_codes: Sequence[str],
    mode: PrecedenceMode | str,
    categories: Sequence[PayrollCategory | str],
    gpt_client: GPTAdjudicator | None = None,
) -> CategoryMappingResponse:
    """Resolve known prior codes deterministically and missing ones through GPT fallback."""

    normalized_prior_codes = [_normalize_prior_code(prior_code) for prior_code in prior_codes]
    resolved_mode = _coerce_mode(mode)
    selected_categories = _normalize_categories(categories)
    results = _empty_result_map()
    missing_by_category = _empty_missing_map(selected_categories)

    for category in selected_categories:
        category_index = index.category_indexes[category]
        for prior_code in normalized_prior_codes:
            if prior_code not in category_index.all_rows:
                missing_by_category[category].append(prior_code)
                continue
            results[category].append(
                _resolve(
                    index=category_index,
                    prior_code=prior_code,
                    mode=resolved_mode,
                )
            )

    if _has_missing(missing_by_category):
        fallback = _resolve_missing_priors_with_gpt(
            missing_prior_codes_by_category=missing_by_category,
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


def _resolve_missing_priors_with_gpt(
    *,
    missing_prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
    index: MappingIndex,
    gpt_client: GPTAdjudicator | None,
) -> CategoryMappingResponse:
    """Resolve missing prior codes using GPT against category-specific catalogs only."""

    requested_categories = tuple(
        category
        for category in PAYROLL_CATEGORIES
        if missing_prior_codes_by_category.get(category)
    )
    catalogs = {
        category: tuple(sorted(index.category_indexes[category].all_global_codes))
        for category in requested_categories
    }
    catalog_evidence = {
        category: index.category_indexes[category].catalog_evidence()
        for category in requested_categories
    }
    missing_pair_count = sum(
        len(prior_codes)
        for prior_codes in missing_prior_codes_by_category.values()
    )
    total_candidates = sum(len(codes) for codes in catalogs.values())
    if total_candidates == 0:
        logger.warning(
            "Missing prior-code fallback has no global-code candidates",
            extra=log_extra("missing_prior_no_candidates", prior_code_count=missing_pair_count),
        )
        return _no_match_response(missing_prior_codes_by_category)

    logger.info(
        "Missing prior-code GPT fallback eligible: priorCodes=%d candidates=%d",
        missing_pair_count,
        total_candidates,
        extra=log_extra(
            "missing_prior_gpt_fallback_eligible",
            prior_code_count=missing_pair_count,
            candidate_count=total_candidates,
            categories=[category.value for category in requested_categories],
        ),
    )

    if gpt_client is None:
        logger.warning(
            "Missing prior-code GPT fallback unavailable; returning NO_MATCH",
            extra=log_extra("missing_prior_gpt_unavailable", prior_code_count=missing_pair_count),
        )
        return _no_match_response(missing_prior_codes_by_category)

    try:
        if len(requested_categories) == 1 and missing_pair_count == 1:
            category = requested_categories[0]
            response = gpt_client.recommend_global_code(
                prior_code=missing_prior_codes_by_category[category][0],
                category=category,
                catalogs=catalogs,
                catalog_evidence=catalog_evidence,
            )
            return _complete_fallback_response(response, missing_prior_codes_by_category)

        response = gpt_client.recommend_global_codes(
            prior_codes_by_category=missing_prior_codes_by_category,
            catalogs=catalogs,
            catalog_evidence=catalog_evidence,
        )
        return _complete_fallback_response(response, missing_prior_codes_by_category)
    except GPTAdjudicationError as exc:
        logger.warning(
            "Missing prior-code GPT fallback failed: %s",
            exc,
            extra=log_extra("missing_prior_gpt_failed", prior_code_count=missing_pair_count),
        )
        return _no_match_response(missing_prior_codes_by_category)
    except Exception:
        logger.exception(
            "Unexpected missing prior-code GPT fallback failure",
            extra=log_extra(
                "missing_prior_gpt_unexpected_failure",
                prior_code_count=missing_pair_count,
            ),
        )
        return _no_match_response(missing_prior_codes_by_category)


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


def _normalize_categories(categories: Sequence[PayrollCategory | str]) -> tuple[PayrollCategory, ...]:
    if not categories:
        raise MappingError("At least one category must be selected")

    selected: set[PayrollCategory] = set()
    for category in categories:
        if isinstance(category, PayrollCategory):
            selected.add(category)
            continue
        try:
            selected.add(PayrollCategory(str(category).strip()))
        except ValueError as exc:
            allowed = [item.value for item in PAYROLL_CATEGORIES]
            raise MappingError(f"Unsupported category '{category}'. Allowed categories: {allowed}") from exc

    return tuple(category for category in PAYROLL_CATEGORIES if category in selected)


def _empty_result_map() -> CategoryResultMap:
    return {category: [] for category in PAYROLL_CATEGORIES}


def _empty_missing_map(
    categories: Sequence[PayrollCategory],
) -> dict[PayrollCategory, list[str]]:
    return {category: [] for category in categories}


def _has_missing(missing_by_category: Mapping[PayrollCategory, Sequence[str]]) -> bool:
    return any(missing_by_category.values())


def _merge_response(target: CategoryResultMap, response: CategoryMappingResponse) -> None:
    for category, items in response.as_category_map().items():
        target[category].extend(items)


def _complete_fallback_response(
    response: CategoryMappingResponse,
    missing_prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
) -> CategoryMappingResponse:
    response_by_category = response.as_category_map()
    completed = _empty_result_map()

    for category in PAYROLL_CATEGORIES:
        expected_prior_codes = [
            _normalize_prior_code(prior_code)
            for prior_code in missing_prior_codes_by_category.get(category, ())
        ]
        by_prior_code = {
            result.priorCode: result
            for result in response_by_category[category]
            if result.priorCode in expected_prior_codes
        }
        completed[category] = [
            by_prior_code.get(
                prior_code,
                MappingResult(priorCode=prior_code, globalCode=NO_MATCH_GLOBAL_CODE),
            )
            for prior_code in expected_prior_codes
        ]

    return CategoryMappingResponse.from_category_map(completed)


def _no_match_response(
    missing_prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
) -> CategoryMappingResponse:
    return CategoryMappingResponse.from_category_map(
        {
            category: [
                MappingResult(priorCode=prior_code, globalCode=NO_MATCH_GLOBAL_CODE)
                for prior_code in missing_prior_codes_by_category.get(category, ())
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
