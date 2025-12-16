import os

import uvicorn
from app.logger import LOGGING_CONFIG

if __name__ == "__main__":
    # Concurrency limit - max simultaneous requests before returning 503
    # This provides backpressure to prevent overload
    def _get_env_as_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    try:
        limit_concurrency = int(os.getenv("VLLM_PROXY_LIMIT_CONCURRENCY", "0")) or None
    except ValueError:
        limit_concurrency = None

    # Connection backlog - queue size for pending connections
    backlog = _get_env_as_int("VLLM_PROXY_BACKLOG", 2048)

    # Keep-alive timeout in seconds
    timeout_keep_alive = _get_env_as_int("VLLM_PROXY_KEEPALIVE_TIMEOUT", 30)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_config=LOGGING_CONFIG,
        log_level="info",
        limit_concurrency=limit_concurrency,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
    )
