"""Optional bounded GPT adjudication layer for tied payroll mappings."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from app.config import PrecedenceMode, settings
from app.exceptions import GPTAdjudicationError
from app.logging_utils import log_extra
from app.prompt_builder import build_missing_prior_prompt

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a payroll code adjudication component for an enterprise banking system.

Select exactly one winner from the provided internal-code candidates.

Output rules:
- Return only a JSON object.
- The JSON object must have exactly one key: "winner".
- The value must be one of the provided candidate codes exactly.
- Do not include explanations, markdown, scores, or new codes.
"""

_MISSING_PRIOR_SYSTEM_PROMPT = """You are a payroll code recommendation component.

You must select one internal payroll code from the provided list, or return NO_MATCH.
Return only the selected code. Do not include explanations, markdown, scores, or new codes.
"""


GptAdjudicationError = GPTAdjudicationError


class GptClient:
    """Thin wrapper around OpenAI or Azure OpenAI chat completions."""

    def __init__(self, client: Any | None = None) -> None:
        self._client: Any | None = client if client is not None else self._build_client()

    def adjudicate(
        self,
        *,
        prior_code: str,
        candidates: list[str],
        mode: PrecedenceMode,
        occurrence_counts: dict[str, int],
        latest_dates: dict[str, str],
    ) -> tuple[str, str]:
        if not self._client:
            raise GPTAdjudicationError("OpenAI client is not configured")
        if not candidates:
            raise GPTAdjudicationError("No tied candidates supplied")

        raw_response = self._call_api(
            system_prompt=_SYSTEM_PROMPT,
            user_message=self._build_user_message(
                prior_code=prior_code,
                candidates=candidates,
                mode=mode,
                occurrence_counts=occurrence_counts,
                latest_dates=latest_dates,
            ),
            response_format_json=True,
        )
        winner = self._parse_response(raw_response, valid_candidates=candidates)
        return winner, raw_response

    def recommend_internal_code(
        self,
        *,
        prior_code: str,
        candidate_codes: Sequence[str],
    ) -> str:
        """Recommend one internal code for a prior code missing from history."""

        if not self._client:
            raise GPTAdjudicationError("OpenAI client is not configured")

        candidates = list(candidate_codes)
        if not candidates:
            raise GPTAdjudicationError("No internal code candidates supplied")

        raw_response = self._call_api(
            system_prompt=_MISSING_PRIOR_SYSTEM_PROMPT,
            user_message=build_missing_prior_prompt(
                prior_code=prior_code,
                candidate_codes=candidates,
            ),
            response_format_json=False,
        )
        return self._parse_plain_code_response(raw_response)

    @staticmethod
    def _build_user_message(
        *,
        prior_code: str,
        candidates: list[str],
        mode: PrecedenceMode,
        occurrence_counts: dict[str, int],
        latest_dates: dict[str, str],
    ) -> str:
        evidence = [
            {
                "internalCode": code,
                "occurrenceCount": occurrence_counts.get(code, 0),
                "latestDate": latest_dates.get(code),
            }
            for code in candidates
        ]
        payload = {
            "priorCode": prior_code,
            "mode": mode.value,
            "candidates": candidates,
            "evidence": evidence,
            "requiredResponseShape": {"winner": "<one candidate code>"},
        }
        return json.dumps(payload, separators=(",", ":"))

    def _call_api(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_format_json: bool,
    ) -> str:
        request: dict[str, object] = {
            "model": settings.effective_openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": settings.openai_max_tokens,
            "temperature": settings.openai_temperature,
        }
        if response_format_json:
            request["response_format"] = {"type": "json_object"}

        logger.info("===== GPT CALL STARTED =====")
        try:
            response = self._client.chat.completions.create(**request)
            logger.info(f"RAW GPT RESPONSE: {response}")
            content = response.choices[0].message.content
        except TypeError as exc:
            if not response_format_json:
                raise GPTAdjudicationError(f"OpenAI API call failed: {exc}") from exc
            request.pop("response_format", None)
            try:
                response = self._client.chat.completions.create(**request)
                logger.info(f"RAW GPT RESPONSE: {response}")
                content = response.choices[0].message.content
            except Exception as retry_exc:
                raise GPTAdjudicationError(
                    f"OpenAI API call failed: {retry_exc}"
                ) from retry_exc
        except Exception as exc:
            raise GPTAdjudicationError(f"OpenAI API call failed: {exc}") from exc

        if not content:
            raise GPTAdjudicationError("OpenAI API returned an empty response")
        return content.strip()

    @staticmethod
    def _parse_plain_code_response(raw_response: str) -> str:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines() if not line.strip().startswith("```")
            ).strip()
        return cleaned.strip().strip('"').strip("'").upper()

    @staticmethod
    def _parse_response(raw_response: str, *, valid_candidates: list[str]) -> str:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines() if not line.strip().startswith("```")
            ).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GPTAdjudicationError(f"GPT returned invalid JSON: {raw_response}") from exc

        if not isinstance(parsed, dict) or set(parsed.keys()) != {"winner"}:
            raise GPTAdjudicationError(f"GPT response must contain only 'winner': {parsed}")

        winner = str(parsed["winner"]).strip().upper()
        if winner not in valid_candidates:
            raise GPTAdjudicationError(
                f"GPT winner '{winner}' is not in tied candidates {valid_candidates}"
            )
        return winner

    @staticmethod
    def _build_client() -> Any | None:
        try:
            from openai import AzureOpenAI, OpenAI  # type: ignore
        except ImportError:
            logger.warning(
                "openai package is not installed; GPT integration disabled",
                extra=log_extra("gpt_client_unavailable"),
            )
            return None

        if settings.uses_azure_openai:
            if not settings.openai_api_key:
                logger.warning(
                    "Azure OpenAI is configured without OPENAI_API_KEY",
                    extra=log_extra("azure_openai_missing_api_key"),
                )
                return None
            logger.info(
                "Azure OpenAI client configured",
                extra=log_extra("gpt_client_configured", provider="azure_openai"),
            )
            return AzureOpenAI(
                api_key=settings.openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )

        if settings.openai_api_key:
            logger.info(
                "OpenAI client configured",
                extra=log_extra("gpt_client_configured", provider="openai"),
            )
            return OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )

        logger.info(
            "No OpenAI API key configured; GPT integration disabled",
            extra=log_extra("gpt_client_disabled"),
        )
        return None
