"""Optional bounded GPT adjudication layer for tied payroll mappings."""

from __future__ import annotations

import json
import logging

from app.config import PrecedenceMode, settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a payroll code adjudication component for an enterprise banking system.

Select exactly one winner from the provided internal-code candidates.

Output rules:
- Return only a JSON object.
- The JSON object must have exactly one key: "winner".
- The value must be one of the provided candidate codes exactly.
- Do not include explanations, markdown, scores, or new codes.
"""


class GptAdjudicationError(RuntimeError):
    """Raised when GPT adjudication fails validation."""


class GptClient:
    """Thin wrapper around OpenAI or Azure OpenAI chat completions."""

    def __init__(self) -> None:
        self._client = self._build_client()

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
            raise GptAdjudicationError("OpenAI client is not configured")
        if not candidates:
            raise GptAdjudicationError("No tied candidates supplied")

        raw_response = self._call_api(
            self._build_user_message(
                prior_code=prior_code,
                candidates=candidates,
                mode=mode,
                occurrence_counts=occurrence_counts,
                latest_dates=latest_dates,
            )
        )
        winner = self._parse_response(raw_response, valid_candidates=candidates)
        return winner, raw_response

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

    def _call_api(self, user_message: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=settings.effective_openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        except TypeError:
            response = self._client.chat.completions.create(
                model=settings.effective_openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise GptAdjudicationError(f"OpenAI API call failed: {exc}") from exc

        if not content:
            raise GptAdjudicationError("OpenAI API returned an empty response")
        return content.strip()

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
            raise GptAdjudicationError(f"GPT returned invalid JSON: {raw_response}") from exc

        if not isinstance(parsed, dict) or set(parsed.keys()) != {"winner"}:
            raise GptAdjudicationError(f"GPT response must contain only 'winner': {parsed}")

        winner = str(parsed["winner"]).strip().upper()
        if winner not in valid_candidates:
            raise GptAdjudicationError(
                f"GPT winner '{winner}' is not in tied candidates {valid_candidates}"
            )
        return winner

    @staticmethod
    def _build_client():
        try:
            from openai import AzureOpenAI, OpenAI  # type: ignore
        except ImportError:
            logger.warning("openai package is not installed; GPT adjudication disabled")
            return None

        if settings.uses_azure_openai:
            if not settings.openai_api_key:
                logger.warning("Azure OpenAI is configured without OPENAI_API_KEY")
                return None
            return AzureOpenAI(
                api_key=settings.openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )

        if settings.openai_api_key:
            return OpenAI(api_key=settings.openai_api_key)

        logger.info("No OpenAI API key configured; GPT adjudication disabled")
        return None
