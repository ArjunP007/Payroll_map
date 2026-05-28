"""Azure integration helpers for Key Vault and Blob Storage."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from app.logging_utils import log_extra

logger = logging.getLogger(__name__)

KEY_VAULT_SECRET_ENV_MAP: Mapping[str, str] = {
    "openai-api-key": "OPENAI_API_KEY",
    "azure-storage-connection-str": "AZURE_STORAGE_CONNECTION_STRING",
}


def load_secrets_from_key_vault(vault_url: str | None = None) -> dict[str, str]:
    """Load configured secrets from Azure Key Vault into environment variables."""

    url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL")
    if not url:
        logger.debug(
            "No Azure Key Vault URL configured",
            extra=log_extra("azure_key_vault_not_configured"),
        )
        return {}

    try:
        from azure.identity import DefaultAzureCredential  # type: ignore
        from azure.keyvault.secrets import SecretClient  # type: ignore
    except ImportError:
        logger.warning(
            "Azure Key Vault packages are not installed",
            extra=log_extra("azure_key_vault_package_missing"),
        )
        return {}

    loaded: dict[str, str] = {}

    try:
        client = SecretClient(vault_url=url, credential=DefaultAzureCredential())
        for secret_name, env_name in KEY_VAULT_SECRET_ENV_MAP.items():
            if os.environ.get(env_name):
                continue
            try:
                value = client.get_secret(secret_name).value
            except Exception as exc:
                logger.warning(
                    "Could not read Key Vault secret '%s': %s",
                    secret_name,
                    exc,
                    extra=log_extra("azure_key_vault_secret_read_failed", secret_name=secret_name),
                )
                continue
            if value:
                os.environ[env_name] = value
                loaded[secret_name] = "***"
        if loaded:
            logger.info(
                "Azure Key Vault secrets loaded",
                extra=log_extra("azure_key_vault_secrets_loaded", secret_count=len(loaded)),
            )
    except Exception as exc:
        logger.error(
            "Could not initialize Azure Key Vault client: %s",
            exc,
            extra=log_extra("azure_key_vault_client_failed"),
        )

    return loaded


def list_dataset_blobs(connection_string: str, container_name: str) -> list[str]:
    """List dataset blob names, returning an empty list on failure."""

    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except ImportError:
        logger.warning(
            "azure-storage-blob is not installed",
            extra=log_extra("azure_blob_package_missing"),
        )
        return []

    try:
        client = BlobServiceClient.from_connection_string(connection_string)
        container = client.get_container_client(container_name)
        return [blob.name for blob in container.list_blobs()]
    except Exception as exc:
        logger.error(
            "Could not list blobs in container '%s': %s",
            container_name,
            exc,
            extra=log_extra("azure_blob_list_failed", container=container_name),
        )
        return []


def blob_exists(connection_string: str, container_name: str, blob_name: str) -> bool:
    """Return whether a dataset blob exists."""

    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except ImportError:
        return False

    try:
        client = BlobServiceClient.from_connection_string(connection_string)
        return client.get_blob_client(container=container_name, blob=blob_name).exists()
    except Exception as exc:
        logger.warning(
            "Could not check blob '%s': %s",
            blob_name,
            exc,
            extra=log_extra("azure_blob_exists_failed", container=container_name, blob=blob_name),
        )
        return False
