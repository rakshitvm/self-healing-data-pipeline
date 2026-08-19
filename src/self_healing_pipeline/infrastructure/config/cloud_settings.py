"""Configuration for the optional Azure Storage / Databricks
integration — deliberately kept out of `settings.py`'s `Settings`
aggregate root (and out of `settings.py` entirely) so that this file can
be added with zero modification to any existing configuration code.

Mirrors `settings.py`'s own `GroqSettings`/`load_groq_settings` pattern
exactly: each class is loaded on demand, only by the separate, optional
`repair-and-process` CLI command — the existing `repair`/`schema-repair`
commands never construct either class, so missing or invalid values here
can never affect them. A missing/invalid `.env` value raises a
`pydantic.ValidationError` at load time; callers are expected to treat
that as "this integration is not configured" and skip it, never as a
reason to fail the repair itself (see `interfaces/cli/main.py`'s
`repair_and_process` command).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTINGS_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


class AzureStorageSettings(BaseSettings):
    """Connection details for uploading a repaired file to Azure Blob
    Storage via a container-scoped SAS token (see
    `infrastructure/cloud/azure_blob_uploader.py`)."""

    model_config = _SETTINGS_CONFIG

    container_url: str = Field(validation_alias="AZURE_STORAGE_CONTAINER_URL")
    sas_token: str = Field(validation_alias="AZURE_STORAGE_SAS_TOKEN")


class DatabricksSettings(BaseSettings):
    """Connection details for triggering a pre-existing Databricks job
    (see `infrastructure/cloud/databricks_job_trigger.py`). Reuses the
    `DATABRICKS_HOST`/`DATABRICKS_TOKEN` variable names already reserved
    in `.env.example`. `notebook_param_name` is the parameter key your
    job's notebook task actually reads — this project has no way to know
    that in advance (it depends entirely on how that notebook was
    authored), so it is configurable with a documented default rather
    than guessed."""

    model_config = _SETTINGS_CONFIG

    host: str = Field(validation_alias="DATABRICKS_HOST")
    token: str = Field(validation_alias="DATABRICKS_TOKEN")
    job_id: int = Field(validation_alias="DATABRICKS_JOB_ID")
    notebook_param_name: str = Field(
        default="input_path", validation_alias="DATABRICKS_NOTEBOOK_PARAM_NAME"
    )


def load_azure_storage_settings() -> AzureStorageSettings:
    """Load `AzureStorageSettings` from `.env`/the environment.

    Raises `pydantic.ValidationError` if `AZURE_STORAGE_CONTAINER_URL`/
    `AZURE_STORAGE_SAS_TOKEN` are not set — callers must catch this and
    treat it as "cloud upload not configured", not as a repair failure.
    """
    return AzureStorageSettings()  # type: ignore[call-arg]


def load_databricks_settings() -> DatabricksSettings:
    """Load `DatabricksSettings` from `.env`/the environment.

    Raises `pydantic.ValidationError` if `DATABRICKS_HOST`/
    `DATABRICKS_TOKEN`/`DATABRICKS_JOB_ID` are not set — callers must
    catch this and treat it as "Databricks trigger not configured", not
    as a repair failure.
    """
    return DatabricksSettings()  # type: ignore[call-arg]
