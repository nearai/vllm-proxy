"""
Backend module for multi-backend vLLM support.

Provides model-based routing, round-robin load balancing, and model discovery.
"""

from .config import BackendConfig, get_backend_config
from .discovery import ModelDiscovery
from .pool import BackendPool, get_backend_pool, init_backend_pool, close_backend_pool

__all__ = [
    "BackendConfig",
    "get_backend_config",
    "ModelDiscovery",
    "BackendPool",
    "get_backend_pool",
    "init_backend_pool",
    "close_backend_pool",
]
