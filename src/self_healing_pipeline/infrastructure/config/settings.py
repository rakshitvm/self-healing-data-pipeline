"""Application configuration.

Defines the configuration layer for the Self-Healing Data Pipeline using
Pydantic Settings (v2). Configuration is grouped into logical sections
(Application, Database, Azure OpenAI, MLflow, Logging), each of which is
loaded from a `.env` file with process environment variables taking
precedence over `.env` values. Required fields have no defaults, so
missing configuration raises a validation error at instantiation time
rather than failing later at the point of use.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTINGS_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


class ApplicationSettings(BaseSettings):
    """Core application identity and runtime environment."""

    model_config = _SETTINGS_CONFIG

    app_name: str = Field(validation_alias="APP_NAME")
    app_env: str = Field(validation_alias="APP_ENV")
    secret_key: str = Field(validation_alias="SECRET_KEY")


class DatabaseSettings(BaseSettings):
    """Connection settings for the primary relational data store."""

    model_config = _SETTINGS_CONFIG

    database_url: str = Field(validation_alias="DATABASE_URL")
    db_host: str = Field(validation_alias="DB_HOST")
    db_port: int = Field(validation_alias="DB_PORT")
    db_name: str = Field(validation_alias="DB_NAME")
    db_user: str = Field(validation_alias="DB_USER")
    db_password: str = Field(validation_alias="DB_PASSWORD")


class AzureOpenAISettings(BaseSettings):
    """Credentials and deployment details for Azure OpenAI."""

    model_config = _SETTINGS_CONFIG

    api_key: str = Field(validation_alias="AZURE_OPENAI_API_KEY")
    endpoint: str = Field(validation_alias="AZURE_OPENAI_ENDPOINT")
    api_version: str = Field(validation_alias="AZURE_OPENAI_API_VERSION")
    deployment_name: str = Field(validation_alias="AZURE_OPENAI_DEPLOYMENT_NAME")


class GroqSettings(BaseSettings):
    """Credentials and model for the temporary Groq development provider.

    Deliberately NOT part of the aggregate `Settings` root below: Groq is
    a temporary development-only stand-in for Azure OpenAI (selected via
    `LLM_PROVIDER=groq`), so its configuration must not become mandatory
    when the system is configured for the production Azure OpenAI
    provider, and vice versa. Loaded on demand by the provider factory
    only when `LLM_PROVIDER=groq` is actually selected.
    """

    model_config = _SETTINGS_CONFIG

    api_key: str = Field(validation_alias="GROQ_API_KEY")
    model: str = Field(validation_alias="GROQ_MODEL")
    base_url: str = Field(
        default="https://api.groq.com/openai/v1", validation_alias="GROQ_BASE_URL"
    )


class MLflowSettings(BaseSettings):
    """Experiment and run tracking configuration for MLflow."""

    model_config = _SETTINGS_CONFIG

    tracking_uri: str = Field(validation_alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(validation_alias="MLFLOW_EXPERIMENT_NAME")


class LoggingSettings(BaseSettings):
    """Logging verbosity and output format configuration."""

    model_config = _SETTINGS_CONFIG

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        validation_alias="LOG_LEVEL"
    )
    log_format: str = Field(validation_alias="LOG_FORMAT")


def _load_application_settings() -> ApplicationSettings:
    # Required fields are populated from the environment at runtime, which
    # mypy cannot verify statically, hence the call-arg suppression.
    return ApplicationSettings()  # type: ignore[call-arg]


def _load_database_settings() -> DatabaseSettings:
    return DatabaseSettings()  # type: ignore[call-arg]


def load_azure_openai_settings() -> AzureOpenAISettings:
    """Load `AzureOpenAISettings`.

    Public — also called directly by the provider factory when
    `LLM_PROVIDER=azure_openai` is selected, in addition to being used as
    the `Settings` aggregate's own default factory below.
    """
    return AzureOpenAISettings()  # type: ignore[call-arg]


def load_groq_settings() -> GroqSettings:
    """Load `GroqSettings` on demand (not part of the `Settings` aggregate).

    Public — unlike the aggregate's private `_load_*` factories, this is
    called directly by the provider factory only when `LLM_PROVIDER=groq`
    is selected, so Groq credentials are never required otherwise.
    """
    return GroqSettings()  # type: ignore[call-arg]


def _load_mlflow_settings() -> MLflowSettings:
    return MLflowSettings()  # type: ignore[call-arg]


def _load_logging_settings() -> LoggingSettings:
    return LoggingSettings()  # type: ignore[call-arg]


class Settings(BaseModel):
    """Aggregate root composing every configuration section.

    Provides a single entry point for application configuration. Each
    section is its own independently validated `BaseSettings` model, so
    constructing `Settings` validates every required field across every
    section in one step.
    """

    application: ApplicationSettings = Field(default_factory=_load_application_settings)
    database: DatabaseSettings = Field(default_factory=_load_database_settings)
    azure_openai: AzureOpenAISettings = Field(default_factory=load_azure_openai_settings)
    mlflow: MLflowSettings = Field(default_factory=_load_mlflow_settings)
    logging: LoggingSettings = Field(default_factory=_load_logging_settings)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated `Settings` instance.

    The first call reads and validates the environment; subsequent calls
    reuse the same instance instead of re-reading `.env`.
    """
    return Settings()
