# Payroll Mapping Engine

Enterprise-style batch payroll code-mapping backend.

The service loads historical nested JSON mappings, normalizes them, builds in-memory indexes, applies a selected precedence rule, and returns only final `priorCode -> internalCode` mappings.

## Project Structure

```text
paycor_mapping/
  app/
    main.py             FastAPI application and routes
    engine.py           Runtime orchestration for loading, indexing, and mapping
    config.py           Environment-backed settings
    exceptions.py       Shared domain exception hierarchy
    schemas.py          Pydantic request, response, and internal models
    loader.py           JSON loading, validation, normalization
    index_builder.py    Fast lookup/index construction
    mapper.py           Precedence and deterministic tie-break logic
    gpt_client.py       Optional bounded GPT adjudication layer
    prompt_builder.py   Constrained GPT prompt construction
    validator.py        Dataset and output validation guards
    azure_storage.py    Azure Key Vault and Blob Storage helpers
    logging_utils.py    Structured logging helpers

  data/
    FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json

  tests/
    test_loader.py
    test_mapper.py
    test_api.py

  deployment/
    Dockerfile
    azure.yaml

  requirements.txt
  pyproject.toml
```

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Main Endpoint

```http
POST /api/v1/map
Content-Type: application/json

{"mode": "MAX_OCCURRENCE"}
```

Response is a strict JSON array only:

```json
[
  {"priorCode": "BASIC_SALARY", "internalCode": "BASIC"},
  {"priorCode": "OVERTIME_PAY", "internalCode": "OT"}
]
```

Supported modes:

- `ONE_TO_ONE`
- `MAX_OCCURRENCE`
- `LAST_MODIFIED_DATE`

Precedence modes are resolved through `MODE_RESOLVERS` in `app/mapper.py`.
Adding a deterministic mode means adding a resolver and registering it with
`register_mode_resolver`.

## Missing Prior-Code Fallback

Known prior codes always use the deterministic precedence engine. GPT is used
only when a single prior-code lookup is missing from the historical dataset.
The fallback sends only the missing prior code and the sorted internal-code
catalog from the dataset, never the full historical JSON.

```http
GET /api/v1/map/REMOTE_HOME_STIPEND?mode=MAX_OCCURRENCE
```

Response shape remains the same mapping object:

```json
{"priorCode": "REMOTE_HOME_STIPEND", "internalCode": "REMOTE_ALLOWANCE"}
```

If GPT is unavailable or returns a code outside the allowed catalog, the service
returns:

```json
{"priorCode": "REMOTE_HOME_STIPEND", "internalCode": "NO_MATCH"}
```

## Configuration

Settings are environment-backed and centralized in `app/config.py`.

Common variables:

- `ENVIRONMENT`: `local`, `development`, `staging`, `production`, or `azure`
- `DATASET_SOURCE`: `local` or `azure`
- `DATASET_LOCAL_PATH`: local JSON path
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_CONTAINER_NAME`
- `AZURE_STORAGE_BLOB_NAME`
- `GPT_ADJUDICATION_ENABLED`
- `GPT_MISSING_PRIOR_FALLBACK_ENABLED`
- `OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `LOG_LEVEL`
- `LOG_JSON`

## Verify

```bash
python -m pytest -q
```

## Deployment

Build the container from the repository root:

```bash
docker build -f deployment/Dockerfile -t payroll-mapping-engine .
```

Azure Container Apps configuration lives in `deployment/azure.yaml`.
