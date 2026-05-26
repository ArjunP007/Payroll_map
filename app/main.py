"""FastAPI entry point for the Payroll Mapping Engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.azure_storage import load_secrets_from_key_vault
from app.config import configure_logging, settings
from app.gpt_client import GptClient
from app.index_builder import MappingIndex, build_index
from app.loader import DatasetLoadError, DatasetSchemaError, load_dataset
from app.mapper import map_all
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    MappingRequest,
    MappingResult,
    ReloadRequest,
    ReloadResponse,
)
from app.validator import ValidationError, validate_loaded_records, validate_mapping_results

configure_logging()
logger = logging.getLogger(__name__)


class AppState:
    """Mutable runtime state replaced on dataset reload."""

    index: MappingIndex | None = None
    gpt_client: GptClient | None = None
    loaded: bool = False
    prior_code_count: int = 0
    record_count: int = 0


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s version=%s", settings.app_name, settings.app_version)
    loaded_secrets = load_secrets_from_key_vault(settings.azure_key_vault_url)
    if loaded_secrets:
        settings.refresh_runtime_secrets_from_env()

    _bootstrap_engine()

    if settings.gpt_adjudication_enabled:
        _state.gpt_client = GptClient()
        logger.info("GPT adjudication is enabled")
    else:
        logger.info("GPT adjudication is disabled")

    yield
    logger.info("Shutting down %s", settings.app_name)


def _bootstrap_engine(source: str | None = None) -> None:
    records = load_dataset(source=source)
    validate_loaded_records(records)
    index = build_index(records)

    _state.index = index
    _state.loaded = True
    _state.prior_code_count = len(index.prior_codes)
    _state.record_count = index.total_records


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Payroll code-mapping engine."
    ),
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            statusCode=422,
        ).model_dump(),
    )


@app.exception_handler(DatasetLoadError)
async def dataset_load_error_handler(request: Request, exc: DatasetLoadError):
    logger.error("Dataset load error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Dataset Unavailable",
            detail="The mapping dataset could not be loaded",
            statusCode=503,
        ).model_dump(),
    )


@app.exception_handler(DatasetSchemaError)
async def dataset_schema_error_handler(request: Request, exc: DatasetSchemaError):
    logger.error("Dataset schema error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Dataset Schema Error",
            detail="The mapping dataset failed server-side validation",
            statusCode=500,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            statusCode=500,
        ).model_dump(),
    )


def _require_engine() -> MappingIndex:
    if not _state.loaded or _state.index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The mapping engine is not ready",
        )
    return _state.index


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/api/v1/health")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Operations"],
)
async def health():
    return HealthResponse(
        status="ok" if _state.loaded else "starting",
        appName=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        datasetLoaded=_state.loaded,
        priorCodeCount=_state.prior_code_count,
        recordCount=_state.record_count,
    )


@app.post(
    "/api/v1/map",
    response_model=list[MappingResult],
    summary="Resolve all prior codes for the selected precedence mode",
    tags=["Mapping"],
)
async def batch_map(request: MappingRequest) -> list[MappingResult]:
    index = _require_engine()
    logger.info("Batch mapping request: mode=%s", request.mode.value)

    results = map_all(index=index, mode=request.mode, gpt_client=_state.gpt_client)
    validate_mapping_results(results, index.prior_codes)
    return results


@app.post(
    "/api/v1/reload",
    response_model=ReloadResponse,
    summary="Reload the dataset and rebuild indexes",
    tags=["Operations"],
)
async def reload_dataset(request: ReloadRequest | None = None):
    source = request.source if request else None
    logger.info("Dataset reload requested: source=%s", source)

    try:
        _bootstrap_engine(source=source)
    except (DatasetLoadError, DatasetSchemaError, ValidationError) as exc:
        logger.error("Dataset reload failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc

    return ReloadResponse(
        status="ok",
        priorCodeCount=_state.prior_code_count,
        recordCount=_state.record_count,
        message=(
            f"Dataset reloaded successfully with {_state.prior_code_count} "
            f"prior codes and {_state.record_count} records"
        ),
    )


@app.get(
    "/api/v1/prior-codes",
    summary="List known prior codes for admin diagnostics",
    tags=["Operations"],
)
async def list_prior_codes() -> dict[str, Any]:
    index = _require_engine()
    return {
        "totalPriorCodes": len(index.prior_codes),
        "priorCodes": list(index.prior_codes),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        log_level=settings.log_level.value.lower(),
    )
