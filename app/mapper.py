"""Deterministic payroll mapping adjudication engine."""

from __future__ import annotations

import logging
from datetime import datetime

from app.config import PrecedenceMode, TieBreakStrategy, settings
from app.index_builder import MappingIndex
from app.schemas import MappingDecisionDetail, MappingResult

logger = logging.getLogger(__name__)


def map_all(
    index: MappingIndex,
    mode: PrecedenceMode | str,
    gpt_client=None,
) -> list[MappingResult]:
    """Resolve every prior code in source order."""

    resolved_mode = _coerce_mode(mode)
    results = [
        _resolve(index=index, prior_code=prior_code, mode=resolved_mode, gpt_client=gpt_client)
        for prior_code in index.prior_codes
    ]
    logger.info("Batch mapping complete: mode=%s mapped=%d", resolved_mode.value, len(results))
    return results


def map_one(
    index: MappingIndex,
    prior_code: str,
    mode: PrecedenceMode | str,
    gpt_client=None,
) -> MappingResult:
    """Resolve one prior code, primarily for tests and admin diagnostics."""

    normalized_prior_code = prior_code.strip().upper()
    if normalized_prior_code not in index.all_rows:
        raise KeyError(f"Prior code '{normalized_prior_code}' not found in the dataset")
    return _resolve(
        index=index,
        prior_code=normalized_prior_code,
        mode=_coerce_mode(mode),
        gpt_client=gpt_client,
    )


def _resolve(
    *,
    index: MappingIndex,
    prior_code: str,
    mode: PrecedenceMode,
    gpt_client,
) -> MappingResult:
    if mode == PrecedenceMode.ONE_TO_ONE:
        winner, detail = _resolve_one_to_one(index, prior_code)
    elif mode == PrecedenceMode.MAX_OCCURRENCE:
        winner, detail = _resolve_max_occurrence(index, prior_code)
    elif mode == PrecedenceMode.LAST_MODIFIED_DATE:
        winner, detail = _resolve_last_modified_date(index, prior_code)
    else:
        raise ValueError(f"Unsupported precedence mode: {mode}")

    if detail.tieBreakApplied and settings.gpt_adjudication_enabled and gpt_client is not None:
        winner, detail = _try_gpt_adjudication(
            gpt_client=gpt_client,
            detail=detail,
            deterministic_winner=winner,
        )

    _log_decision(detail)
    return MappingResult(priorCode=prior_code, internalCode=winner)


def _resolve_one_to_one(index: MappingIndex, prior_code: str) -> tuple[str, MappingDecisionDetail]:
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

    detail = _detail(
        index=index,
        prior_code=prior_code,
        mode=PrecedenceMode.ONE_TO_ONE,
        winner=winner,
        primary_rule="single_unique_internal_code",
        secondary_rule=secondary_rule,
        tied_candidates=tied_candidates,
        tie_break_applied=tie_break_applied,
        tie_strategy=tie_strategy,
    )
    return winner, detail


def _resolve_max_occurrence(
    index: MappingIndex, prior_code: str
) -> tuple[str, MappingDecisionDetail]:
    winner, tied_candidates, tie_break_applied, tie_strategy = _rank_candidates(
        index=index,
        prior_code=prior_code,
        candidates=sorted(index.unique_codes[prior_code]),
        ranking=[
            ("occurrence_count", _count_metric(index, prior_code)),
            ("latest_date", _date_metric(index, prior_code)),
        ],
    )
    detail = _detail(
        index=index,
        prior_code=prior_code,
        mode=PrecedenceMode.MAX_OCCURRENCE,
        winner=winner,
        primary_rule="highest_occurrence_count",
        secondary_rule="latest_date_when_count_ties",
        tied_candidates=tied_candidates,
        tie_break_applied=tie_break_applied,
        tie_strategy=tie_strategy,
    )
    return winner, detail


def _resolve_last_modified_date(
    index: MappingIndex, prior_code: str
) -> tuple[str, MappingDecisionDetail]:
    winner, tied_candidates, tie_break_applied, tie_strategy = _rank_candidates(
        index=index,
        prior_code=prior_code,
        candidates=sorted(index.unique_codes[prior_code]),
        ranking=[
            ("latest_date", _date_metric(index, prior_code)),
            ("occurrence_count", _count_metric(index, prior_code)),
        ],
    )
    detail = _detail(
        index=index,
        prior_code=prior_code,
        mode=PrecedenceMode.LAST_MODIFIED_DATE,
        winner=winner,
        primary_rule="most_recent_last_modified_date",
        secondary_rule="occurrence_count_when_date_ties",
        tied_candidates=tied_candidates,
        tie_break_applied=tie_break_applied,
        tie_strategy=tie_strategy,
    )
    return winner, detail


def _rank_candidates(
    *,
    index: MappingIndex,
    prior_code: str,
    candidates: list[str],
    ranking: list[tuple[str, dict[str, int | datetime]]],
) -> tuple[str, list[str], bool, str | None]:
    """Apply primary and secondary metrics, then hidden deterministic tie-break."""

    remaining = list(candidates)
    if not remaining:
        raise RuntimeError(f"No candidates available for prior code '{prior_code}'")

    for _rule_name, metric_values in ranking:
        best_value = max(metric_values[code] for code in remaining)
        remaining = [code for code in remaining if metric_values[code] == best_value]
        if len(remaining) == 1:
            return remaining[0], remaining, False, None

    winner = _break_tie(index=index, prior_code=prior_code, candidates=remaining)
    return winner, sorted(remaining), True, _tie_break_strategy().value


def _break_tie(index: MappingIndex, prior_code: str, candidates: list[str]) -> str:
    strategy = _tie_break_strategy()
    if strategy == TieBreakStrategy.LEXICOGRAPHIC:
        return sorted(candidates)[0]

    first_seen = index.first_seen_order[prior_code]
    return sorted(candidates, key=lambda code: (first_seen.get(code, 10**9), code))[0]


def _count_metric(index: MappingIndex, prior_code: str) -> dict[str, int]:
    return dict(index.occurrence_counts[prior_code])


def _date_metric(index: MappingIndex, prior_code: str) -> dict[str, datetime]:
    return dict(index.latest_dates[prior_code])


def _detail(
    *,
    index: MappingIndex,
    prior_code: str,
    mode: PrecedenceMode,
    winner: str,
    primary_rule: str,
    secondary_rule: str | None,
    tied_candidates: list[str],
    tie_break_applied: bool,
    tie_strategy: str | None,
) -> MappingDecisionDetail:
    latest_dates = {
        code: date.strftime("%Y-%m-%d")
        for code, date in index.latest_dates[prior_code].items()
    }
    return MappingDecisionDetail(
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


def _try_gpt_adjudication(
    *,
    gpt_client,
    detail: MappingDecisionDetail,
    deterministic_winner: str,
) -> tuple[str, MappingDecisionDetail]:
    try:
        gpt_winner, raw_response = gpt_client.adjudicate(
            prior_code=detail.priorCode,
            candidates=detail.tiedCandidates,
            mode=detail.mode,
            occurrence_counts=detail.occurrenceCounts,
            latest_dates=detail.latestDates,
        )
    except Exception as exc:
        logger.warning(
            "GPT adjudication failed for priorCode=%s: %s. Using deterministic winner=%s",
            detail.priorCode,
            exc,
            deterministic_winner,
        )
        return deterministic_winner, detail

    if gpt_winner not in detail.tiedCandidates:
        logger.warning(
            "GPT returned invalid winner=%s for priorCode=%s. Using deterministic winner=%s",
            gpt_winner,
            detail.priorCode,
            deterministic_winner,
        )
        return deterministic_winner, detail

    updated = detail.model_copy(
        update={
            "winningCode": gpt_winner,
            "gptAdjudicated": True,
            "gptRawResponse": raw_response,
        }
    )
    return gpt_winner, updated


def _coerce_mode(mode: PrecedenceMode | str) -> PrecedenceMode:
    if isinstance(mode, PrecedenceMode):
        return mode
    return PrecedenceMode(str(mode).strip().upper())


def _tie_break_strategy() -> TieBreakStrategy:
    value = settings.tie_break_strategy
    if isinstance(value, TieBreakStrategy):
        return value
    try:
        return TieBreakStrategy(str(value).strip().lower())
    except ValueError:
        return TieBreakStrategy.FIRST_SEEN


def _log_decision(detail: MappingDecisionDetail) -> None:
    logger.debug(
        "MAP_DECISION priorCode=%s mode=%s winner=%s candidates=%d unique=%d "
        "primary=%s secondary=%s tie=%s gpt=%s",
        detail.priorCode,
        detail.mode.value,
        detail.winningCode,
        detail.candidateCount,
        len(detail.uniqueCandidates),
        detail.primaryRule,
        detail.secondaryRule,
        detail.tieBreakApplied,
        detail.gptAdjudicated,
    )
