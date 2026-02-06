# CLAUDE.md - Project Guide for Claude Code

## Project Overview

vllm-proxy is a FastAPI-based proxy service for vLLM backends that adds:
- Request/response signing with ECDSA and Ed25519
- Optional end-to-end encryption for sensitive endpoints
- TEE (Trusted Execution Environment) attestation support
- Signature caching and retrieval via `/v1/signature/{id}`

## Running Tests

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run the main API tests (most reliable)
pytest tests/app/test_openai.py -v

# Run a specific test
pytest tests/app/test_openai.py::test_chat_completions_with_request_hash_non_streaming -v

# Run tests matching a pattern
pytest tests/app/test_openai.py -k "streaming" -v
```

### Known Test Issues

- **4 pre-existing failures** in `test_openai.py` related to oversized request error format assertions (tests expect `{"error": ...}` but get `{"detail": ...}`)
- **Encryption tests** (`test_*_encryption.py`) require GPU dependencies (`pynvml`) that may not be available locally
- **`tests/app/encryption/test_encryption.py`** has import errors due to missing `pynvml` module

## Project Structure

```
src/app/
├── api/v1/openai.py    # Main API endpoints (chat, completions, images, etc.)
├── cache/cache.py      # Signature caching (requires MODEL_NAME env var)
├── encryption/         # E2E encryption helpers
├── quote/quote.py      # Signing contexts (ECDSA, Ed25519), attestation
└── logger.py           # Logging configuration
```

## Key Patterns

### Signature Flow
1. Calculate request hash: `sha256(request_body)` or use `X-Request-Hash` header
2. Forward request to vLLM backend
3. Extract/generate response `id` (inject if missing)
4. Calculate response hash: `sha256(response_body)`
5. Sign: `sign_chat(f"{req_hash}:{res_hash}")`
6. Cache: `cache.set_chat(id, signature_json)`
7. Client retrieves via: `GET /v1/signature/{id}`

### ID Generation Prefixes
- `chatcmpl-` for chat completions
- `img-` for image generations
- `emb-` for embeddings
- `trans-` for transcriptions
- `rerank-` for reranking
- `score-` for scoring
- `passthrough-` for passthrough endpoint

### Encryption Headers
Both required to enable encryption:
- `X-Signing-Algo`: `ecdsa` or `ed25519`
- `X-Client-Pub-Key`: Client's public key in hex

## Environment Variables

- `VLLM_BASE_URL` - Backend vLLM service URL (default: `http://vllm:8000`)
- `MODEL_NAME` - Required for cache initialization
- `VLLM_PROXY_MAX_REQUEST_SIZE` - Max request body size (default: 10MB)
- `VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE` - Max image request size (default: 50MB)
- `VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE` - Max audio request size (default: 100MB)

## Dependencies

Managed via Poetry. Key dependencies:
- `fastapi` with `uvicorn`
- `httpx` for async HTTP client
- `web3`, `eth-account` for crypto signing
- `pynacl` for Ed25519
- `redis` for caching
