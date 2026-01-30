"""
Backend pool for multi-backend vLLM support.

Provides round-robin load balancing with model-based routing.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.logger import log

from .config import BackendConfig, get_backend_config
from .discovery import ModelDiscovery


@dataclass
class BackendPool:
    """
    Manages a pool of backend servers with round-robin load balancing.

    Supports model-based routing where different backends may serve different models.
    """

    config: BackendConfig
    discovery: ModelDiscovery = field(default_factory=ModelDiscovery)
    _round_robin_counters: dict[str, int] = field(default_factory=dict)
    _counter_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize discovery TTL from config."""
        self.discovery.ttl = self.config.model_discovery_ttl

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        """Set the HTTP client for the discovery component."""
        self.discovery.set_http_client(client)

    async def select_backend(
        self, model: str, exclude_backends: Optional[list[str]] = None
    ) -> Optional[str]:
        """
        Select a backend for a given model using round-robin.

        Args:
            model: The model ID requested
            exclude_backends: List of backend URLs to exclude (e.g., failed backends)

        Returns:
            Backend URL or None if no backend serves the model
        """
        exclude_backends = exclude_backends or []

        # Get backends that serve this model
        backends = await self.discovery.get_backends_for_model(
            model, self.config.urls
        )

        # Filter out excluded backends
        available_backends = [b for b in backends if b not in exclude_backends]

        if not available_backends:
            return None

        # Round-robin selection
        async with self._counter_lock:
            counter_key = model
            if counter_key not in self._round_robin_counters:
                self._round_robin_counters[counter_key] = 0

            index = self._round_robin_counters[counter_key] % len(available_backends)
            self._round_robin_counters[counter_key] += 1

        return available_backends[index]

    async def get_all_backends_for_model(self, model: str) -> list[str]:
        """
        Get all backends that serve a specific model.

        Args:
            model: The model ID to look for

        Returns:
            List of backend URLs that serve the model
        """
        return await self.discovery.get_backends_for_model(model, self.config.urls)

    def build_url(self, backend_url: str, path: str) -> str:
        """
        Construct full URL for a backend endpoint.

        Args:
            backend_url: The base backend URL (e.g., http://vllm:8000)
            path: The endpoint path (e.g., /v1/chat/completions)

        Returns:
            Full URL (e.g., http://vllm:8000/v1/chat/completions)
        """
        return f"{backend_url.rstrip('/')}{path}"

    async def get_all_models(self) -> list[dict]:
        """
        Get aggregated list of all models from all backends.

        Returns:
            List of model objects with 'id' field
        """
        return await self.discovery.get_all_models(self.config.urls)


# Global pool instance
_backend_pool: Optional[BackendPool] = None


def init_backend_pool() -> BackendPool:
    """
    Initialize the global backend pool.

    Should be called during application startup.
    """
    global _backend_pool
    config = get_backend_config()
    _backend_pool = BackendPool(config=config)
    log.info(
        f"Backend pool initialized with {len(config.urls)} backend(s): {config.urls}"
    )
    return _backend_pool


def get_backend_pool() -> BackendPool:
    """
    Get the global backend pool instance.

    Raises:
        RuntimeError: If the pool hasn't been initialized
    """
    global _backend_pool
    if _backend_pool is None:
        raise RuntimeError(
            "Backend pool not initialized. Call init_backend_pool() first."
        )
    return _backend_pool


async def close_backend_pool() -> None:
    """
    Close the backend pool and clean up resources.

    Should be called during application shutdown.
    """
    global _backend_pool
    if _backend_pool is not None:
        _backend_pool.discovery.invalidate_cache()
        _backend_pool = None
        log.info("Backend pool closed")


def reset_backend_pool() -> None:
    """Reset the global backend pool. Used for testing."""
    global _backend_pool
    _backend_pool = None
