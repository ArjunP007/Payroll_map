"""Runtime orchestration for dataset loading, indexing, and mapping."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.config import DatasetSource, PrecedenceMode, settings
from app.exceptions import EngineNotReadyError
from app.gpt_client import GptClient
from app.index_builder import MappingIndex, build_index
from app.loader import load_dataset
from app.logging_utils import log_extra
from app.mapper import GPTAdjudicator, map_all as resolve_all
from app.schemas import MappingResult
from app.validator import validate_loaded_records, validate_mapping_results

logger = logging.getLogger(__name__)

GptClientFactory = Callable[[], GPTAdjudicator | None]


@dataclass(frozen=True)
class EngineSnapshot:
    """Small immutable view of engine readiness for API responses."""

    loaded: bool
    prior_code_count: int
    record_count: int


class PayrollMappingEngine:
    """Application service that keeps FastAPI routes free of mapping internals."""

    def __init__(self, gpt_client_factory: GptClientFactory | None = None) -> None:
        self._index: MappingIndex | None = None
        self._gpt_client: GPTAdjudicator | None = None
        self._gpt_client_factory = gpt_client_factory or GptClient

    @property
    def snapshot(self) -> EngineSnapshot:
        index = self._index
        return EngineSnapshot(
            loaded=index is not None,
            prior_code_count=len(index.prior_codes) if index is not None else 0,
            record_count=index.total_records if index is not None else 0,
        )

    def initialize(self, source: str | DatasetSource | None = None) -> None:
        """Load the dataset, build indexes, and configure optional GPT support."""

        self.reload(source=source)
        self.configure_gpt_adjudication()

    def reload(self, source: str | DatasetSource | None = None) -> EngineSnapshot:
        """Reload records and atomically replace the active in-memory index."""

        records = load_dataset(source=source)
        validate_loaded_records(records)
        index = build_index(records)

        self._index = index
        snapshot = self.snapshot
        logger.info(
            "Mapping engine dataset ready: priorCodes=%d records=%d",
            snapshot.prior_code_count,
            snapshot.record_count,
            extra=log_extra(
                "engine_dataset_ready",
                prior_code_count=snapshot.prior_code_count,
                record_count=snapshot.record_count,
            ),
        )
        return snapshot

    def configure_gpt_adjudication(self) -> None:
        """Configure the optional GPT adjudicator without making it mandatory."""

        if not settings.gpt_adjudication_enabled:
            self._gpt_client = None
            logger.info(
                "GPT adjudication is disabled",
                extra=log_extra("gpt_adjudication_disabled"),
            )
            return

        try:
            self._gpt_client = self._gpt_client_factory()
        except Exception:
            self._gpt_client = None
            logger.exception(
                "GPT adjudication setup failed; deterministic mapping remains available",
                extra=log_extra("gpt_adjudication_setup_failed"),
            )
            return

        logger.info(
            "GPT adjudication is enabled",
            extra=log_extra("gpt_adjudication_enabled"),
        )

    def map_all(self, mode: PrecedenceMode | str) -> list[MappingResult]:
        """Resolve all prior codes and validate the strict public result shape."""

        index = self.require_index()
        results = resolve_all(index=index, mode=mode, gpt_client=self._gpt_client)
        validate_mapping_results(results, index.prior_codes)
        return results

    def prior_codes(self) -> tuple[str, ...]:
        """Return known prior codes in source order."""

        return self.require_index().prior_codes

    def require_index(self) -> MappingIndex:
        """Return the active index or raise a domain-specific readiness error."""

        if self._index is None:
            raise EngineNotReadyError("The mapping engine is not ready")
        return self._index
