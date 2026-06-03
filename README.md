# Payroll Mapping Engine

Enterprise-style batch payroll code-mapping backend.

The service loads category-scoped historical JSON mappings, normalizes them, builds in-memory indexes per payroll namespace, applies a selected precedence rule, and returns final `priorCode -> globalCode` mappings grouped by Earnings, Deductions, and Taxes.

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
    gpt_client.py       Optional bounded GPT missing-prior fallback layer
    prompt_builder.py   Constrained GPT prompt construction
    validator.py        Dataset and output validation guards
    azure_storage.py    Azure Key Vault and Blob Storage helpers
    logging_utils.py    Structured logging helpers

  data/
    FULL_50PC_250GC_PRECEDENCE_STRESS_DATASET.json

  tests/
    test_loader.py
    test_mapper.py
    test_gpt_prompt.py
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

## Dataset Shape

The canonical dataset contract is category-scoped:

```json
{
  "Earnings": {
    "BASIC_SALARY": [
      {
        "globalCode": "BASIC",
        "LastModifiedDate": "05/20/2025",
        "payType": "salary",
        "country": "US"
      }
    ]
  },
  "Deductions": {
    "HEALTH_INSURANCE": [
      {
        "globalCode": "INS",
        "LastModifiedDate": "05/18/2025"
      }
    ]
  },
  "Taxes": {
    "INCOME_TAX": [
      {
        "globalCode": "TAX",
        "LastModifiedDate": "05/17/2025"
      }
    ]
  }
}
```

Deterministic mapping uses only `priorCode`, `globalCode`, and `LastModifiedDate`. Extra row fields are preserved as metadata for GPT fallback only.

## Main Endpoint

```http
POST /api/v1/map
Content-Type: application/json

{"mode": "MAX_OCCURRENCE", "categories": ["Earnings", "Deductions", "Taxes"]}
```

Response is strict EDT-grouped JSON:

```json
{
  "Earnings": [
    {"priorCode": "BASIC_SALARY", "globalCode": "BASIC"},
    {"priorCode": "OVERTIME_PAY", "globalCode": "OT"}
  ],
  "Deductions": [
    {"priorCode": "HEALTH_INSURANCE", "globalCode": "INS"}
  ],
  "Taxes": [
    {"priorCode": "INCOME_TAX", "globalCode": "TAX"}
  ]
}
```

Supported modes:

- `ONE_TO_ONE`
- `MAX_OCCURRENCE`
- `LAST_MODIFIED_DATE`

Precedence modes are resolved through `MODE_RESOLVERS` in `app/mapper.py`.
Adding a deterministic mode means adding a resolver and registering it with
`register_mode_resolver`.

## Single And Batch Lookup

Known prior codes always use the deterministic precedence engine within each category namespace.
Category selection is required and is represented as enum-backed multi-select input in Swagger.

```http
GET /api/v1/map/REMOTE_HOME_STIPEND?selectedCategories=Earnings&selectedCategories=Taxes&mode=MAX_OCCURRENCE
```

Selected prior codes can be resolved in one request:

```http
POST /api/v1/map/batch
Content-Type: application/json

{
  "mode": "MAX_OCCURRENCE",
  "categories": ["Earnings", "Deductions", "Taxes"],
  "priorCodes": ["ADVANCE_RECOVERY", "REMOTE_HOME_STIPEND"]
}
```

GPT is used only when a requested prior code is missing in a selected category. The fallback receives only the selected category catalogs, the missing prior code list by category, and metadata-rich candidate evidence when available. GPT output is validated and must use the same EDT shape:

```json
{
  "Earnings": [
    {"priorCode": "REMOTE_HOME_STIPEND", "globalCode": "REMOTE_ALLOWANCE"}
  ],
  "Deductions": [],
  "Taxes": []
}
```

If GPT is unavailable, uncertain, or returns a code outside the category catalog, the service returns `NO_MATCH`.

## Configuration

Settings are environment-backed and centralized in `app/config.py`.

Common variables:

- `ENVIRONMENT`: `local`, `development`, `staging`, `production`, or `azure`
- `DATASET_SOURCE`: `local` or `azure`
- `DATASET_LOCAL_PATH`: local JSON path
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_CONTAINER_NAME`
- `AZURE_STORAGE_BLOB_NAME`
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
