"""
Model discovery for multi-backend vLLM support.

Handles lazy discovery of models from backend servers with caching.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.logger import log


@dataclass
class BackendModelInfo:
    """Cached model information for a backend."""

    models: list[str]
    timestamp: float


@dataclass
class ModelDiscovery:
    """
    Discovers and caches which models are served by each backend.

    Uses lazy discovery - queries each backend's /v1/models endpoint
    on first request, then caches with configurable TTL.
    """

    ttl: int = 60  # Cache TTL in seconds
    _cache: dict[str, BackendModelInfo] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _http_client: Optional[httpx.AsyncClient] = None

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        """Set the HTTP client for making requests."""
        self._http_client = client

    def _get_lock(self, backend_url: str) -> asyncio.Lock:
        """Get or create a lock for a backend URL to prevent thundering herd."""
        if backend_url not in self._locks:
            self._locks[backend_url] = asyncio.Lock()
        return self._locks[backend_url]

    def _is_cache_valid(self, backend_url: str) -> bool:
        """Check if the cache for a backend is still valid."""
        if backend_url not in self._cache:
            return False
        info = self._cache[backend_url]
        return (time.time() - info.timestamp) < self.ttl

    async def discover_models(self, backend_url: str) -> list[str]:
        """
        Discover models served by a backend.

        Uses cached data if available and not expired, otherwise queries
        the backend's /v1/models endpoint.

        Args:
            backend_url: The base URL of the backend (e.g., http://vllm:8000)

        Returns:
            List of model IDs served by the backend
        """
        # Return cached data if valid
        if self._is_cache_valid(backend_url):
            return self._cache[backend_url].models

        # Acquire lock to prevent thundering herd
        lock = self._get_lock(backend_url)
        async with lock:
            # Double-check cache after acquiring lock
            if self._is_cache_valid(backend_url):
                return self._cache[backend_url].models

            # Query the backend
            models = await self._fetch_models(backend_url)
            self._cache[backend_url] = BackendModelInfo(
                models=models,
                timestamp=time.time(),
            )
            return models

    async def _fetch_models(self, backend_url: str) -> list[str]:
        """Fetch models from a backend's /v1/models endpoint."""
        if self._http_client is None:
            log.error("HTTP client not set for model discovery")
            return []

        url = f"{backend_url.rstrip('/')}/v1/models"
        try:
            response = await self._http_client.get(url, timeout=10.0)
            if response.status_code != 200:
                log.warning(
                    f"Failed to fetch models from {url}: "
                    f"status={response.status_code}"
                )
                return []

            data = response.json()
            models = []
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if "id" in item:
                        models.append(item["id"])
            log.info(f"Discovered models from {backend_url}: {models}")
            return models
        except Exception as e:
            log.warning(f"Failed to fetch models from {url}: {type(e).__name__}")
            return []

    async def get_backends_for_model(
        self, model: str, backend_urls: list[str]
    ) -> list[str]:
        """
        Get list of backends that serve a specific model.

        Args:
            model: The model ID to look for
            backend_urls: List of backend URLs to check

        Returns:
            List of backend URLs that serve the model
        """
        matching_backends = []

        # Discover models from all backends concurrently
        tasks = [self.discover_models(url) for url in backend_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for backend_url, models_or_error in zip(backend_urls, results):
            if isinstance(models_or_error, Exception):
                log.warning(
                    f"Error discovering models from {backend_url}: "
                    f"{type(models_or_error).__name__}"
                )
                continue
            models = models_or_error
            if model in models:
                matching_backends.append(backend_url)

        return matching_backends

    def invalidate_cache(self, backend_url: Optional[str] = None) -> None:
        """
        Invalidate the model cache.

        Args:
            backend_url: If provided, invalidate only this backend's cache.
                        If None, invalidate all caches.
        """
        if backend_url:
            self._cache.pop(backend_url, None)
        else:
            self._cache.clear()

    async def get_all_models(self, backend_urls: list[str]) -> list[dict]:
        """
        Get aggregated list of all models from all backends.

        Args:
            backend_urls: List of backend URLs to query

        Returns:
            List of model objects with 'id' field
        """
        seen_models = set()
        all_models = []

        tasks = [self.discover_models(url) for url in backend_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for backend_url, models_or_error in zip(backend_urls, results):
            if isinstance(models_or_error, Exception):
                continue
            models = models_or_error
            for model_id in models:
                if model_id not in seen_models:
                    seen_models.add(model_id)
                    all_models.append({"id": model_id, "object": "model"})

        return all_models
