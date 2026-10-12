"""
Security and authentication module placeholder.

Provides API key and token extraction hooks for future secure multi-tenant access.
"""

from typing import Optional
from fastapi import Header, HTTPException, status


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Optional API key validation dependency.
    For local development, unrestricted access is allowed when not configured.
    """
    # In production, validate against AWS Secrets Manager / KMS or API Gateway Cognito authorizer
    return x_api_key
