"""
Backend configuration for multi-backend vLLM support.
"""

import os
from dataclasses import dataclass
from typing import Optional

_backend_config: Optional["BackendConfig"] = None


@dataclass
class BackendConfig:
    """Configuration for backend connections."""

    urls: list[str]
    model_discovery_ttl: int
    is_multi_backend: bool

    @property
    def single_url(self) -> str:
        """Get the single backend URL (for backward compatibility)."""
        return self.urls[0] if self.urls else "http://vllm:8000"


def get_backend_config() -> BackendConfig:
    """
    Get backend configuration from environment variables.

    Prioritizes VLLM_BASE_URLS over VLLM_BASE_URL for backward compatibility.
    """
    global _backend_config
    if _backend_config is not None:
        return _backend_config

    # Check for multi-backend configuration first
    vllm_base_urls = os.getenv("VLLM_BASE_URLS", "").strip()

    if vllm_base_urls:
        # Multi-backend mode: parse comma-separated URLs
        urls = [url.strip() for url in vllm_base_urls.split(",") if url.strip()]
        is_multi_backend = len(urls) > 1
    else:
        # Single backend mode (backward compatible)
        single_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000").strip()
        urls = [single_url]
        is_multi_backend = False

    # Model discovery TTL (default 60 seconds)
    model_discovery_ttl = int(os.getenv("MODEL_DISCOVERY_TTL", "60"))

    _backend_config = BackendConfig(
        urls=urls,
        model_discovery_ttl=model_discovery_ttl,
        is_multi_backend=is_multi_backend,
    )
    return _backend_config


def reset_backend_config() -> None:
    """Reset the cached backend configuration. Used for testing."""
    global _backend_config
    _backend_config = None
