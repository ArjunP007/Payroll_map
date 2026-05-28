"""FastAPI entry point for the Payroll Mapping Engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.azure_storage import load_secrets_from_key_vault
from app.config import configure_logging, settings
from app.engine import PayrollMappingEngine
from app.exceptions import (
    DatasetLoadError,
    DatasetSchemaError,
    EngineNotReadyError,
    MappingError,
    ValidationError,
)
from app.logging_utils import log_extra
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    MappingRequest,
    MappingResult,
    PriorCodesResponse,
    ReloadRequest,
    ReloadResponse,
)

configure_logging()
logger = logging.getLogger(__name__)

engine = PayrollMappingEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s version=%s environment=%s",
        settings.app_name,
        settings.app_version,
        settings.environment.value,
        extra=log_extra(
            "service_starting",
            app_name=settings.app_name,
            app_version=settings.app_version,
            environment=settings.environment.value,
        ),
    )
    loaded_secrets = load_secrets_from_key_vault(settings.azure_key_vault_url)
    if loaded_secrets:
        settings.refresh_runtime_secrets_from_env()

    engine.initialize()

    yield
    logger.info(
        "Shutting down %s",
        settings.app_name,
        extra=log_extra("service_stopping", app_name=settings.app_name),
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Payroll code-mapping engine.",
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
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        extra=log_extra("validation_error", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            statusCode=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ).model_dump(),
    )


@app.exception_handler(EngineNotReadyError)
async def engine_not_ready_handler(request: Request, exc: EngineNotReadyError) -> JSONResponse:
    logger.warning(
        "Engine not ready on %s %s",
        request.method,
        request.url.path,
        extra=log_extra("engine_not_ready", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Service Unavailable",
            detail=exc.public_message,
            statusCode=status.HTTP_503_SERVICE_UNAVAILABLE,
        ).model_dump(),
    )


@app.exception_handler(DatasetLoadError)
async def dataset_load_error_handler(request: Request, exc: DatasetLoadError) -> JSONResponse:
    logger.error(
        "Dataset load error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        extra=log_extra("dataset_load_error", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Dataset Unavailable",
            detail=exc.public_message,
            statusCode=status.HTTP_503_SERVICE_UNAVAILABLE,
        ).model_dump(),
    )


@app.exception_handler(DatasetSchemaError)
async def dataset_schema_error_handler(request: Request, exc: DatasetSchemaError) -> JSONResponse:
    logger.error(
        "Dataset schema error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        extra=log_extra("dataset_schema_error", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Dataset Schema Error",
            detail=exc.public_message,
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


@app.exception_handler(MappingError)
async def mapping_error_handler(request: Request, exc: MappingError) -> JSONResponse:
    logger.error(
        "Mapping error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        extra=log_extra("mapping_error", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Mapping Error",
            detail=exc.public_message,
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        extra=log_extra("unhandled_exception", path=request.url.path),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/api/v1/health")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    snapshot = engine.snapshot
    return HealthResponse(
        status="ok" if snapshot.loaded else "starting",
        appName=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
        datasetLoaded=snapshot.loaded,
        priorCodeCount=snapshot.prior_code_count,
        recordCount=snapshot.record_count,
    )


@app.post(
    "/api/v1/map",
    response_model=list[MappingResult],
    summary="Resolve all prior codes for the selected precedence mode",
    tags=["Mapping"],
)
async def batch_map(request: MappingRequest) -> list[MappingResult]:
    logger.info(
        "Batch mapping request: mode=%s",
        request.mode.value,
        extra=log_extra("mapping_request", mode=request.mode.value),
    )
    return engine.map_all(request.mode)


@app.post(
    "/api/v1/reload",
    response_model=ReloadResponse,
    summary="Reload the dataset and rebuild indexes",
    tags=["Operations"],
)
async def reload_dataset(request: ReloadRequest | None = None) -> ReloadResponse:
    source = request.source if request else None
    logger.info(
        "Dataset reload requested: source=%s",
        source.value if source else None,
        extra=log_extra("dataset_reload_requested", source=source.value if source else None),
    )

    snapshot = engine.reload(source=source)
    return ReloadResponse(
        status="ok",
        priorCodeCount=snapshot.prior_code_count,
        recordCount=snapshot.record_count,
        message=(
            f"Dataset reloaded successfully with {snapshot.prior_code_count} "
            f"prior codes and {snapshot.record_count} records"
        ),
    )


@app.get(
    "/api/v1/prior-codes",
    response_model=PriorCodesResponse,
    summary="List known prior codes for admin diagnostics",
    tags=["Operations"],
)
async def list_prior_codes() -> PriorCodesResponse:
    prior_codes = list(engine.prior_codes())
    return PriorCodesResponse(
        totalPriorCodes=len(prior_codes),
        priorCodes=prior_codes,
    )


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
