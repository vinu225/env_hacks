"""
Application Domain Exceptions.

Provides a unified hierarchy of exceptions for the platform, ensuring clear
separation between domain errors, provider timeouts, and internal model issues.
"""

from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Base exception for all application-specific errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class EntityNotFoundError(BaseAppException):
    """Raised when an entity (location, report, prediction) does not exist."""

    def __init__(self, entity_name: str, entity_id: str):
        super().__init__(
            message=f"{entity_name} with id '{entity_id}' was not found.",
            code="ENTITY_NOT_FOUND",
            status_code=404,
            details={"entity_name": entity_name, "entity_id": entity_id},
        )


class ProviderError(BaseAppException):
    """Raised when an external provider (weather, Jev, LLM) encounters an error."""

    def __init__(self, provider_name: str, message: str, original_error: Optional[str] = None):
        super().__init__(
            message=f"Provider '{provider_name}' error: {message}",
            code="PROVIDER_ERROR",
            status_code=502,
            details={"provider": provider_name, "original_error": original_error},
        )


class ModelInferenceError(BaseAppException):
    """Raised when a risk prediction model fails during inference."""

    def __init__(self, model_name: str, model_version: str, reason: str):
        super().__init__(
            message=f"Model '{model_name}:{model_version}' failed during inference: {reason}",
            code="MODEL_INFERENCE_ERROR",
            status_code=500,
            details={"model_name": model_name, "model_version": model_version, "reason": reason},
        )


class FeatureComputationError(BaseAppException):
    """Raised when feature transformation fails due to missing or corrupt observations."""

    def __init__(self, feature_name: str, reason: str):
        super().__init__(
            message=f"Failed to compute feature '{feature_name}': {reason}",
            code="FEATURE_COMPUTATION_ERROR",
            status_code=422,
            details={"feature_name": feature_name, "reason": reason},
        )
