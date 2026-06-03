"""Optional bounded GPT fallback layer for missing payroll mappings."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.config import PAYROLL_CATEGORIES, PayrollCategory, settings
from app.exceptions import GPTAdjudicationError
from app.logging_utils import log_extra
from app.prompt_builder import (
    NO_MATCH_GLOBAL_CODE,
    build_missing_prior_batch_prompt,
    build_missing_prior_prompt,
)
from app.schemas import CategoryMappingResponse

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a payroll code recommendation component.

Return only strict JSON matching the requested schema. Do not include explanations, markdown,
scores, confidence, or extra keys. Never invent codes.
"""

GptAdjudicationError = GPTAdjudicationError


class GptClient:
    """Thin wrapper around OpenAI or Azure OpenAI chat completions."""

    def __init__(self, client: Any | None = None) -> None:
        self._client: Any | None = client if client is not None else self._build_client()

    def recommend_global_code(
        self,
        *,
        prior_code: str,
        category: PayrollCategory,
        catalogs: Mapping[PayrollCategory, Sequence[str]],
        catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, Any]]],
    ) -> CategoryMappingResponse:
        """Recommend category-scoped global-code mappings for one missing prior code."""

        if not self._client:
            raise GPTAdjudicationError("OpenAI client is not configured")

        raw_response = self._call_api(
            system_prompt=_SYSTEM_PROMPT,
            user_message=build_missing_prior_prompt(
                prior_code=prior_code,
                category=category,
                catalogs=catalogs,
                catalog_evidence=catalog_evidence,
            ),
            response_format_json=True,
        )
        return self._parse_category_response(raw_response, allowed_catalogs=catalogs)

    def recommend_global_codes(
        self,
        *,
        prior_codes_by_category: Mapping[PayrollCategory, Sequence[str]],
        catalogs: Mapping[PayrollCategory, Sequence[str]],
        catalog_evidence: Mapping[PayrollCategory, Sequence[Mapping[str, Any]]],
    ) -> CategoryMappingResponse:
        """Recommend category-scoped global-code mappings for missing prior codes."""

        if not self._client:
            raise GPTAdjudicationError("OpenAI client is not configured")
        if not any(prior_codes_by_category.values()):
            raise GPTAdjudicationError("No missing prior codes supplied")

        raw_response = self._call_api(
            system_prompt=_SYSTEM_PROMPT,
            user_message=build_missing_prior_batch_prompt(
                prior_codes_by_category=prior_codes_by_category,
                catalogs=catalogs,
                catalog_evidence=catalog_evidence,
            ),
            response_format_json=True,
        )
        return self._parse_category_response(raw_response, allowed_catalogs=catalogs)

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
            logger.info("RAW GPT RESPONSE: %s", response)
            content = response.choices[0].message.content
        except TypeError as exc:
            if not response_format_json:
                raise GPTAdjudicationError(f"OpenAI API call failed: {exc}") from exc
            request.pop("response_format", None)
            try:
                response = self._client.chat.completions.create(**request)
                logger.info("RAW GPT RESPONSE: %s", response)
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
    def _parse_category_response(
        raw_response: str,
        *,
        allowed_catalogs: Mapping[PayrollCategory, Sequence[str]],
    ) -> CategoryMappingResponse:
        cleaned = _strip_code_fences(raw_response)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GPTAdjudicationError(f"GPT returned invalid JSON: {raw_response}") from exc

        expected_keys = {category.value for category in PAYROLL_CATEGORIES}
        if not isinstance(parsed, dict) or set(parsed.keys()) != expected_keys:
            raise GPTAdjudicationError(
                f"GPT response must contain exactly category keys {sorted(expected_keys)}"
            )

        allowed_by_category = {
            category: {str(code).strip().upper() for code in allowed_catalogs.get(category, ())}
            for category in PAYROLL_CATEGORIES
        }
        payload: dict[str, list[dict[str, str]]] = {}
        for category in PAYROLL_CATEGORIES:
            raw_items = parsed[category.value]
            if not isinstance(raw_items, list):
                raise GPTAdjudicationError(f"GPT category '{category.value}' must be a list")

            items: list[dict[str, str]] = []
            for raw_item in raw_items:
                if not isinstance(raw_item, dict) or set(raw_item.keys()) != {"priorCode", "globalCode"}:
                    raise GPTAdjudicationError(
                        "GPT mapping items must contain exactly priorCode and globalCode"
                    )
                prior_code = str(raw_item["priorCode"]).strip().upper()
                global_code = str(raw_item["globalCode"]).strip().upper()
                if not prior_code:
                    raise GPTAdjudicationError("GPT returned an empty priorCode")
                if global_code != NO_MATCH_GLOBAL_CODE and global_code not in allowed_by_category[category]:
                    raise GPTAdjudicationError(
                        f"GPT globalCode '{global_code}' is not allowed for {category.value}"
                    )
                items.append({"priorCode": prior_code, "globalCode": global_code})
            payload[category.value] = items

        return CategoryMappingResponse.model_validate(payload)

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


def _strip_code_fences(raw_response: str) -> str:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()
    return cleaned
