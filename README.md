# Payroll Mapping Engine

Enterprise-style batch payroll code-mapping backend.

The service loads historical nested JSON mappings, normalizes them, builds in-memory indexes, applies a selected precedence rule, and returns only final `priorCode -> internalCode` mappings.

## Project Structure

```text
paycor_mapping/
  app/
    main.py             FastAPI application and routes
    config.py           Environment-backed settings
    schemas.py          Pydantic request, response, and internal models
    loader.py           JSON loading, validation, normalization
    index_builder.py    Fast lookup/index construction
    mapper.py           Precedence and deterministic tie-break logic
    gpt_client.py       Optional bounded GPT adjudication layer
    validator.py        Dataset and output validation guards
    azure_storage.py    Azure Key Vault and Blob Storage helpers

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
