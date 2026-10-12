"""
Application Configuration Module.

Uses Pydantic Settings to load and strictly validate environment variables.
Supports twelve-factor app principles and local/staging/production profiles.
"""

from functools import lru_cache
from typing import List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core metadata
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = "development"
    APP_NAME: str = "Environmental-Intelligence-Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # Integration Providers (Pluggable: mock for local/test, real providers for cloud)
    WEATHER_PROVIDER_TYPE: Literal["mock", "open_meteo", "custom"] = "mock"
    DECISION_PROVIDER_TYPE: Literal["mock", "jev"] = "mock"
    LLM_PROVIDER_TYPE: Literal["mock", "bedrock", "openai"] = "mock"

    # ML Model Configuration
    HEAT_MODEL_TYPE: Literal["ml", "rule_based"] = "rule_based"
    HEAT_MODEL_ARTIFACT_DIR: str = ""

    # External Provider Configs
    OPEN_METEO_API_URL: str = "https://api.open-meteo.com/v1"
    JEV_API_ENDPOINT: str = ""
    JEV_API_KEY: str = ""
    AWS_BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"

    # Storage Abstraction (InMemory for test/local; DynamoDB/S3 for AWS cloud)
    STORAGE_BACKEND: Literal["memory", "dynamodb"] = "memory"
    DYNAMODB_TABLE_LOCATIONS: str = "bharat-env-locations"
    DYNAMODB_TABLE_RISKS: str = "bharat-env-risk-predictions"
    DYNAMODB_TABLE_REPORTS: str = "bharat-env-community-reports"
    S3_BUCKET_ARTIFACTS: str = "bharat-env-ml-artifacts"

    # AWS Deployment Settings
    AWS_REGION: str = "ap-south-1"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
