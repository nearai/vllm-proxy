import os

import uvicorn
from app.logger import LOGGING_CONFIG

if __name__ == "__main__":
    # Concurrency limit - max simultaneous requests before returning 503
    # This provides backpressure to prevent overload
    try:
        limit_concurrency = int(os.getenv("VLLM_PROXY_LIMIT_CONCURRENCY", "0")) or None
    except ValueError:
        limit_concurrency = None

    # Connection backlog - queue size for pending connections
    try:
        backlog = int(os.getenv("VLLM_PROXY_BACKLOG", "2048"))
    except ValueError:
        backlog = 2048

    # Keep-alive timeout in seconds
    try:
        timeout_keep_alive = int(os.getenv("VLLM_PROXY_KEEPALIVE_TIMEOUT", "30"))
    except ValueError:
        timeout_keep_alive = 30

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
