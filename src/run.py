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

    # Server port - can be overridden via environment variable
    port = _get_env_as_int("VLLM_PROXY_PORT", 8000)

    # TLS configuration - enable HTTPS when certificate and key are provided
    ssl_certfile = os.getenv("VLLM_PROXY_SSL_CERTFILE")
    ssl_keyfile = os.getenv("VLLM_PROXY_SSL_KEYFILE")
    ssl_keyfile_password = os.getenv("VLLM_PROXY_SSL_KEYFILE_PASSWORD")
    ssl_ca_certs = os.getenv("VLLM_PROXY_SSL_CA_CERTS")

    # Build SSL kwargs only if TLS is configured
    ssl_kwargs = {}
    if ssl_certfile and ssl_keyfile:
        ssl_kwargs["ssl_certfile"] = ssl_certfile
        ssl_kwargs["ssl_keyfile"] = ssl_keyfile
        if ssl_keyfile_password:
            ssl_kwargs["ssl_keyfile_password"] = ssl_keyfile_password
        if ssl_ca_certs:
            ssl_kwargs["ssl_ca_certs"] = ssl_ca_certs

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_config=LOGGING_CONFIG,
        log_level="info",
        limit_concurrency=limit_concurrency,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        **ssl_kwargs,
    )
