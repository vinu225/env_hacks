"""
Repository Base Abstraction.

Decouples domain business logic from specific persistence engines
(DynamoDB, S3, PostgreSQL, In-Memory).
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")
ID = TypeVar("ID")


class BaseRepository(ABC, Generic[T, ID]):
    """Abstract base repository contract."""

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> Optional[T]:
        """Fetch entity by primary key."""
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100, skip: int = 0) -> List[T]:
        """List entities with basic pagination."""
        pass

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Create or update entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete entity by ID."""
        pass
