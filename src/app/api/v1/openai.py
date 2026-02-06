import json
import os
import uuid
from hashlib import sha256
from typing import Optional

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from app.api.helper.auth import verify_authorization_header
from app.api.response.response import (
    invalid_signing_algo,
    not_found,
    unexpect_error,
)
from app.cache.cache import cache
from app.encryption.encryption import (
    decrypt_data,
    encrypt_data,
)
from app.logger import log
from app.quote.quote import (
    ECDSA,
    ED25519,
    SigningContext,
    ecdsa_context,
    ed25519_context,
    generate_attestation,
    sign_message,
)

router = APIRouter(tags=["openai"])

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000")
VLLM_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/completions"
VLLM_TOKENIZE_URL = f"{VLLM_BASE_URL}/tokenize"
VLLM_METRICS_URL = f"{VLLM_BASE_URL}/metrics"
VLLM_MODELS_URL = f"{VLLM_BASE_URL}/v1/models"
# Image generation endpoint - can be overridden to point to a different service
VLLM_IMAGES_URL = os.getenv("VLLM_IMAGES_URL", f"{VLLM_BASE_URL}/v1/images/generations")
# Image edit endpoint - can be overridden to point to a different service
VLLM_IMAGES_EDITS_URL = os.getenv(
    "VLLM_IMAGES_EDITS_URL", f"{VLLM_BASE_URL}/v1/images/edits"
)
# Audio transcription endpoint - can be overridden to point to a different service
VLLM_TRANSCRIPTIONS_URL = os.getenv(
    "VLLM_TRANSCRIPTIONS_URL", f"{VLLM_BASE_URL}/v1/audio/transcriptions"
)
VLLM_EMBEDDINGS_URL = f"{VLLM_BASE_URL}/v1/embeddings"
# Reranking endpoint - can be overridden to point to a different service
VLLM_RERANK_URL = os.getenv("VLLM_RERANK_URL", f"{VLLM_BASE_URL}/v1/rerank")
# Score endpoint - can be overridden to point to a different service
VLLM_SCORE_URL = os.getenv("VLLM_SCORE_URL", f"{VLLM_BASE_URL}/v1/score")
TIMEOUT = 60 * 10
TIMEOUT_TOKENIZE = 10

COMMON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


# Connection pool limits - tune based on expected load
# max_connections: total concurrent connections to vLLM backend
# max_keepalive_connections: connections kept alive for reuse
def _get_env_as_int(name: str, default: int) -> int:
    """Safely gets an environment variable as an integer, falling back to a default."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


MAX_CONNECTIONS = _get_env_as_int("VLLM_PROXY_MAX_CONNECTIONS", 1000)
MAX_KEEPALIVE_CONNECTIONS = _get_env_as_int("VLLM_PROXY_MAX_KEEPALIVE", 100)

# Backend TLS verification settings
# VLLM_PROXY_BACKEND_SSL_VERIFY: Set to "false" to disable SSL verification (not recommended for production)
# VLLM_PROXY_BACKEND_SSL_CA_BUNDLE: Path to CA bundle for verifying backend certificates
_backend_ssl_verify_env = os.getenv("VLLM_PROXY_BACKEND_SSL_VERIFY", "true").lower()
BACKEND_SSL_VERIFY: bool | str = _backend_ssl_verify_env not in ("false", "0", "no")
_backend_ca_bundle = os.getenv("VLLM_PROXY_BACKEND_SSL_CA_BUNDLE")
if _backend_ca_bundle and BACKEND_SSL_VERIFY:
    # Use custom CA bundle path instead of boolean True
    BACKEND_SSL_VERIFY = _backend_ca_bundle

# Maximum request body size (default 10MB) - prevents memory exhaustion attacks
# Critical for TEE environments with limited memory
MAX_REQUEST_SIZE = _get_env_as_int("VLLM_PROXY_MAX_REQUEST_SIZE", 10 * 1024 * 1024)

# Maximum request body size for image requests (default 50MB per OpenAI docs)
# Larger limit needed for base64-encoded images
MAX_IMAGE_REQUEST_SIZE = _get_env_as_int(
    "VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE", 50 * 1024 * 1024
)

# Maximum request body size for audio/multimodal requests (default 100MB)
# Larger limit needed for base64-encoded audio in Qwen3-Omni multimodal requests
MAX_AUDIO_REQUEST_SIZE = _get_env_as_int(
    "VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE", 100 * 1024 * 1024
)

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        limits = httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        )
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(TIMEOUT),
            headers=COMMON_HEADERS,
            limits=limits,
            verify=BACKEND_SSL_VERIFY,
        )
    return _http_client


async def close_http_client():
    """Close the shared HTTP client. Call on app shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def read_body_with_limit(
    request: Request, max_size: int = MAX_REQUEST_SIZE
) -> bytes:
    """
    Read request body with size limit to prevent memory exhaustion attacks.

    This is critical for TEE environments where memory is limited and non-expandable.
    The function first checks the Content-Length header for early rejection, then
    streams the body chunk-by-chunk with running size validation.

    Args:
        request: The FastAPI Request object
        max_size: Maximum allowed body size in bytes (default from MAX_REQUEST_SIZE)

    Returns:
        The request body as bytes

    Raises:
        HTTPException: 413 Payload Too Large if body exceeds max_size
    """
    # Check Content-Length header first for early rejection
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum: {max_size} bytes",
                )
        except ValueError:
            # Invalid Content-Length header, continue to stream validation
            pass

    # Stream the body with size validation
    chunks: list[bytes] = []
    total_size = 0
    async for chunk in request.stream():
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Maximum: {max_size} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def read_upload_file_with_limit(file: UploadFile, max_size: int) -> bytes:
    """
    Read an UploadFile with size limit to prevent memory exhaustion attacks.

    This is critical for TEE environments where memory is limited and non-expandable.
    The function first checks the file.size attribute for early rejection (if available
    from Content-Length), then streams the file chunk-by-chunk with running size validation.

    Args:
        file: The FastAPI UploadFile object
        max_size: Maximum allowed file size in bytes

    Returns:
        The file content as bytes

    Raises:
        HTTPException: 413 Payload Too Large if file exceeds max_size
    """
    # Check file.size attribute first for early rejection (set from Content-Length)
    if file.size is not None and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {max_size} bytes",
        )

    # Stream the file with size validation
    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 64 * 1024  # 64KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum: {max_size} bytes",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def sign_request(request: dict, response: str):
    content = json.dumps(request.get("messages", [])) + "\n" + response
    return quote.sign(content)


def hash(payload: str):
    return sha256(payload.encode()).hexdigest()


def sign_chat(text: str):
    return dict(
        text=text,
        signature_ecdsa=sign_message(ecdsa_context, text),
        signing_address_ecdsa=ecdsa_context.signing_address,
        signature_ed25519=sign_message(ed25519_context, text),
        signing_address_ed25519=ed25519_context.signing_address,
    )


def validate_encryption_headers(
    x_signing_algo: Optional[str], x_client_pub_key: Optional[str]
) -> SigningContext:
    """
    Validate encryption headers and return the signing context for decryption.

    Args:
        x_signing_algo: Signing algorithm ('ecdsa' or 'ed25519')
        x_client_pub_key: Client's public key in hex format

    Returns:
        SigningContext for the specified algorithm

    Raises:
        HTTPException: If validation fails (invalid algorithm, invalid key format, etc.)
    """
    # Validate signing algorithm
    if x_signing_algo not in [ECDSA, ED25519]:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid X-Signing-Algo. Must be "{ECDSA}" or "{ED25519}"',
        )

    # Validate public key format and length
    if not x_client_pub_key:
        raise HTTPException(status_code=400, detail="X-Client-Pub-Key cannot be empty")

    # Check if it's a valid hex string
    try:
        public_key_bytes = bytes.fromhex(x_client_pub_key)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="X-Client-Pub-Key must be a valid hex string"
        )

    # Validate length based on algorithm
    if x_signing_algo == ED25519:
        # Ed25519: 32 bytes = 64 hex characters
        if len(public_key_bytes) != 32:
            raise HTTPException(
                status_code=400,
                detail=f"Ed25519 public key must be 64 hex characters (32 bytes), got {len(x_client_pub_key)} characters",
            )
    elif x_signing_algo == ECDSA:
        # ECDSA: 64 bytes (128 hex chars) or 65 bytes with 0x04 prefix (130 hex chars)
        if len(public_key_bytes) == 65 and public_key_bytes[0] == 0x04:
            # Uncompressed format with 0x04 prefix - valid
            pass
        elif len(public_key_bytes) == 64:
            # Uncompressed format without 0x04 prefix - valid
            pass
        else:
            raise HTTPException(
                status_code=400,
                detail=f"ECDSA public key must be 128 hex characters (64 bytes) or 130 hex characters (65 bytes with 0x04 prefix), got {len(x_client_pub_key)} characters",
            )

    # Get and return the signing context for decryption
    return ecdsa_context if x_signing_algo == ECDSA else ed25519_context


def _decrypt_field(field_value: str, context: SigningContext) -> str:
    """
    Decrypt a field value if it's encrypted (hex string).
    Returns decrypted value or raise an exception if decryption fails.
    """
    if len(field_value) == 0:
        return field_value

    try:
        # Try to decode as hex and decrypt
        encrypted_data = bytes.fromhex(field_value)
        decrypted_content = decrypt_data(encrypted_data, context)
        return decrypted_content.decode("utf-8")
    except Exception as e:
        log.error(f"Failed to decrypt field: {type(e).__name__}")
        raise HTTPException(status_code=400, detail="Failed to decrypt field")


def decrypt_message_content(message: dict, context: SigningContext) -> dict:
    """
    Decrypt the content field of a message if it's encrypted.
    Handles both plain text and multimodal content (JSON arrays).
    """
    message = dict(message)  # Create a copy to avoid mutating the original

    if (
        "content" in message
        and message["content"] is not None
        and isinstance(message["content"], str)
        and message["content"]
    ):
        decrypted = _decrypt_field(message["content"], context)
        try:
            parsed = json.loads(decrypted)
            if isinstance(parsed, list):
                message["content"] = parsed
                return message
        except json.JSONDecodeError:
            # Not JSON, so it's plain text.
            pass

        # If it wasn't a JSON list, assign the original decrypted string.
        message["content"] = decrypted

    return message


def encrypt_message_content(
    message: dict, client_public_key: str, signing_algo: str
) -> dict:
    """
    Encrypt the content, reasoning content, and audio fields of a message.
    Handles both plain text and multimodal content (JSON arrays).
    Also handles Qwen3-Omni audio output in message.audio.data field.
    """
    message = dict(message)  # Create a copy to avoid mutating the original

    for field in ["content", "reasoning_content", "reasoning"]:
        if field in message and message[field] is not None and message[field]:
            content = message[field]
            if isinstance(content, list):
                content = json.dumps(content)
            if isinstance(content, str):
                try:
                    encrypted_data = encrypt_data(
                        content.encode("utf-8"), client_public_key, signing_algo
                    )
                    message[field] = encrypted_data.hex()
                except Exception as e:
                    log.error(f"Failed to encrypt message {field}: {type(e).__name__}")
                    raise HTTPException(
                        status_code=500, detail=f"Failed to encrypt message {field}"
                    )

    # Handle Qwen3-Omni audio output: encrypt message.audio.data field
    if "audio" in message:
        audio = message.get("audio")
        if not isinstance(audio, dict):
            raise HTTPException(
                status_code=400, detail="Invalid audio field: expected object"
            )
        audio_data = audio.get("data")
        if isinstance(audio_data, str) and audio_data:
            try:
                encrypted_data = encrypt_data(
                    audio_data.encode("utf-8"), client_public_key, signing_algo
                )
                # Create copy with encrypted data to avoid mutating original
                message["audio"] = {**audio, "data": encrypted_data.hex()}
            except Exception as e:
                log.error(f"Failed to encrypt message audio.data: {type(e).__name__}")
                raise HTTPException(
                    status_code=500, detail="Failed to encrypt message audio.data"
                )

    return message


def decrypt_prompt(prompt, context: SigningContext):
    """
    Decrypt the prompt field if it's encrypted.
    Prompt can be a string or array of strings.
    """
    if isinstance(prompt, str):
        return _decrypt_field(prompt, context)
    elif isinstance(prompt, list):
        return [_decrypt_field(p, context) if isinstance(p, str) else p for p in prompt]
    return prompt


def encrypt_text(text: str, client_public_key: str, signing_algo: str) -> str:
    """
    Encrypt the text field in completions response.
    Returns encrypted text as hex string.
    """
    # Handle None or empty string
    if text is None or text == "":
        log.debug("encrypt_text skipping empty/None value")
        return text

    # Convert to string if not already (e.g., bytes)
    if not isinstance(text, str):
        log.warning(
            f"encrypt_text received non-string type: {type(text).__name__}, "
            f"attempting conversion"
        )
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        else:
            text = str(text)

    try:
        text_bytes = text.encode("utf-8")
        log.debug(f"encrypt_text: input length={len(text)}, bytes length={len(text_bytes)}")
        encrypted_data = encrypt_data(text_bytes, client_public_key, signing_algo)
        result = encrypted_data.hex()
        log.debug(f"encrypt_text: output length={len(result)}")
        return result
    except Exception as e:
        log.error(f"Failed to encrypt text: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Failed to encrypt text")


def encrypt_embedding(
    embedding: list, client_public_key: str, signing_algo: str
) -> object:
    """
    Encrypt an embedding vector.
    Serializes the float array to JSON, encrypts, and returns a hex string.
    If the input is not a non-empty list, it is returned unmodified.
    """
    if not isinstance(embedding, list) or not embedding:
        return embedding

    try:
        # Serialize embedding to JSON string
        embedding_json = json.dumps(embedding)
        encrypted_data = encrypt_data(
            embedding_json.encode("utf-8"), client_public_key, signing_algo
        )
        return encrypted_data.hex()
    except Exception as e:
        log.error(f"Failed to encrypt embedding: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Failed to encrypt embedding")


def decrypt_rerank_documents(documents: list, context: SigningContext) -> list:
    """
    Decrypt documents for reranking.
    Documents can be strings or objects with a 'text' field.
    """
    decrypted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            # Document is a plain string - decrypt it
            decrypted_docs.append(_decrypt_field(doc, context))
        elif isinstance(doc, dict) and "text" in doc:
            # Document is an object with 'text' field - decrypt the text
            doc_copy = dict(doc)
            doc_copy["text"] = _decrypt_field(doc["text"], context)
            decrypted_docs.append(doc_copy)
        else:
            # Unknown format - pass through unchanged
            decrypted_docs.append(doc)
    return decrypted_docs


async def stream_vllm_response(
    url: str,
    request_body: bytes,
    modified_request_body: bytes,
    request_hash: Optional[str] = None,
    encrypt_response: bool = False,
    client_public_key: Optional[str] = None,
    signing_algo: Optional[str] = None,
):
    """
    Handle streaming vllm request
    Args:
        url: The vllm backend URL
        request_body: The original request body
        modified_request_body: The modified enhanced request body
        request_hash: Optional hash from request header (X-Request-Hash). Used by trusted clients to provide
                     pre-calculated request hash, avoiding redundant hash computation. Falls back to
                     calculating hash from request_body if not provided
        encrypt_response: Whether to encrypt the response chunks
        client_public_key: Client's public key for encryption (required if encrypt_response=True)
        signing_algo: Signing algorithm for encryption (required if encrypt_response=True)
    Returns:
        A streaming response (encrypted if encrypt_response=True)
    """
    if request_hash:
        request_sha256 = request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    chat_id = None
    h = sha256()

    async def generate_stream(response):
        nonlocal chat_id, h
        try:
            async for chunk in response.aiter_text():
                data = chunk.strip("data: ").strip()

                # Extract the cache key (data.id) from the first chunk
                if not chat_id:
                    if not data or data == "[DONE]":
                        h.update(chunk.encode())
                        yield chunk
                        continue
                    try:
                        chunk_data = json.loads(data)
                        chat_id = chunk_data.get("id")
                    except Exception as e:
                        error_message = (
                            f"Failed to parse the first chunk: {type(e).__name__}"
                        )
                        log.error(error_message)
                        raise Exception(error_message)

                # Encrypt only message content if needed
                if encrypt_response and client_public_key and signing_algo:
                    # Skip encryption for empty or done chunks
                    if not data or data == "[DONE]":
                        h.update(chunk.encode())
                        yield chunk
                        continue

                    try:
                        chunk_data = json.loads(data)

                        # Note: The 'modality' field (used by Qwen3-Omni for audio streaming)
                        # is preserved at the chunk level since we only modify choices[].delta
                        # and choices[].text, not the top-level chunk fields.

                        # Encrypt content in choices[].delta or choices[].text (for completions)
                        if "choices" in chunk_data:
                            for choice in chunk_data["choices"]:
                                # Handle chat completions format: delta.message.content
                                if (
                                    "delta" in choice
                                    and choice["delta"] is not None
                                    and choice["delta"]
                                ):
                                    choice["delta"] = encrypt_message_content(
                                        choice["delta"], client_public_key, signing_algo
                                    )
                                # Handle completions format: text field
                                elif (
                                    "text" in choice
                                    and choice["text"] is not None
                                    and choice["text"]
                                ):
                                    choice["text"] = encrypt_text(
                                        choice["text"], client_public_key, signing_algo
                                    )

                        # Create the modified chunk string
                        modified_chunk = f"data: {json.dumps(chunk_data)}\n\n"
                        # Hash the encrypted chunk (what client receives)
                        h.update(modified_chunk.encode())
                        # Yield the encrypted chunk
                        yield modified_chunk
                    except Exception as e:
                        error_message = (
                            f"Failed to encrypt chunk content: {type(e).__name__}"
                        )
                        log.error(error_message)
                        raise Exception(error_message)
                else:
                    # Hash and yield plain chunk
                    h.update(chunk.encode())
                    yield chunk

            response_sha256 = h.hexdigest()
            # Cache the full request and response using the extracted cache key
            if chat_id:
                cache.set_chat(
                    chat_id,
                    json.dumps(sign_chat(f"{request_sha256}:{response_sha256}")),
                )
            else:
                error_message = "Chat id could not be extracted from the response"
                log.error(error_message)
                raise Exception(error_message)
        finally:
            await response.aclose()

    client = get_http_client()

    # Forward the request to the vllm backend
    req = client.build_request("POST", url, content=modified_request_body)
    response = await client.send(req, stream=True)
    # If not 200, return the error response directly without streaming
    if response.status_code != 200:
        error_content = await response.aread()
        await response.aclose()

        return Response(
            content=error_content,
            status_code=response.status_code,
            headers=response.headers,
        )

    return StreamingResponse(
        generate_stream(response),
        media_type="text/event-stream",
    )


# Function to handle non-streaming responses
async def non_stream_vllm_response(
    url: str,
    request_body: bytes,
    modified_request_body: bytes,
    request_hash: Optional[str] = None,
    encrypt_response: bool = False,
    client_public_key: Optional[str] = None,
    signing_algo: Optional[str] = None,
):
    """
    Handle non-streaming responses
    Args:
        request_body: The original request body
        modified_request_body: The modified enhanced request body
        request_hash: Optional hash from request header (X-Request-Hash). Used by trusted clients to provide
                     pre-calculated request hash, avoiding redundant hash computation. Falls back to
                     calculating hash from request_body if not provided
        encrypt_response: Whether to encrypt the response content
        client_public_key: Client's public key for encryption (required if encrypt_response=True)
        signing_algo: Signing algorithm for encryption (required if encrypt_response=True)
    Returns:
        The response data (with encrypted content if encrypt_response=True)
    """
    if request_hash:
        request_sha256 = request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    client = get_http_client()
    response = await client.post(url, content=modified_request_body)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response content
    if encrypt_response and client_public_key and signing_algo:
        if "choices" in response_data:
            for choice in response_data["choices"]:
                # Handle chat completions format: message.content
                if (
                    "message" in choice
                    and choice["message"] is not None
                    and choice["message"]
                ):
                    choice["message"] = encrypt_message_content(
                        choice["message"], client_public_key, signing_algo
                    )
                # Handle completions format: text field
                elif "text" in choice and choice["text"] is not None and choice["text"]:
                    choice["text"] = encrypt_text(
                        choice["text"], client_public_key, signing_algo
                    )

    # Cache the request-response pair using the chat ID
    chat_id = response_data.get("id")
    if not chat_id:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        response_data["id"] = chat_id
        log.debug(f"Generated chat ID: {chat_id}")

    # Hash the encrypted response (what client receives) for easier verification
    # Serialize the encrypted response_data to bytes for hashing
    encrypted_response_body = json.dumps(
        response_data, separators=(",", ":")
    ).encode("utf-8")
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        chat_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return response_data


def strip_empty_tool_calls(payload: dict) -> dict:
    """
    Strip empty tool calls from the payload
    To fix the bug of:
    https://github.com/vllm-project/vllm/pull/14054
    """
    if "messages" not in payload:
        return payload

    filtered_messages = []
    for message in payload["messages"]:
        # If the message has tool_calls, filter out empty ones
        if (
            "tool_calls" in message
            and isinstance(message["tool_calls"], list)
            and len(message["tool_calls"]) == 0
        ):
            del message["tool_calls"]
        filtered_messages.append(message)

    payload["messages"] = filtered_messages
    return payload


# Get attestation report of intel quote and nvidia payload
@router.get("/attestation/report", dependencies=[Depends(verify_authorization_header)])
async def attestation_report(
    request: Request,
    signing_algo: str | None = None,
    nonce: str | None = Query(None),
    signing_address: str | None = Query(None),
):
    signing_algo = ECDSA if signing_algo is None else signing_algo
    if signing_algo not in [ECDSA, ED25519]:
        return invalid_signing_algo()

    context = ecdsa_context if signing_algo == ECDSA else ed25519_context

    # If signing_address is specified and doesn't match this server's address, return 404
    if signing_address and context.signing_address.lower() != signing_address.lower():
        raise HTTPException(
            status_code=404, detail="Signing address not found on this server"
        )
    try:
        attestation = generate_attestation(context, nonce)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resp = dict(attestation)
    resp["all_attestations"] = [attestation]
    return resp


# VLLM Chat completions
@router.post("/chat/completions", dependencies=[Depends(verify_authorization_header)])
async def chat_completions(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Chat completions endpoint with optional end-to-end encryption.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request message content should be encrypted as hex strings
    - Response message content will be encrypted as hex strings
    - Only message content is encrypted, not the entire request/response body
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use larger size limit to support audio/multimodal content (audio_url in messages)
    request_body = await read_body_with_limit(request, max_size=MAX_AUDIO_REQUEST_SIZE)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt message content if encryption is enabled
    if encrypt_enabled and "messages" in request_json:
        decrypted_messages = []
        for message in request_json["messages"]:
            try:
                decrypted_message = decrypt_message_content(message, context)
                decrypted_messages.append(decrypted_message)
            except HTTPException:
                raise
            except Exception as e:
                log.error(f"Failed to decrypt message content: {type(e).__name__}")
                raise HTTPException(
                    status_code=400, detail="Failed to decrypt message content"
                )
        request_json["messages"] = decrypted_messages

    modified_json = strip_empty_tool_calls(request_json)

    # Check if the request is for streaming or non-streaming
    is_stream = modified_json.get(
        "stream", False
    )  # Default to non-streaming if not specified

    modified_request_body = json.dumps(modified_json).encode("utf-8")

    if is_stream:
        # Create a streaming response
        return await stream_vllm_response(
            VLLM_URL,
            request_body,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_client_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
    else:
        # Handle non-streaming response
        response_data = await non_stream_vllm_response(
            VLLM_URL,
            request_body,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_client_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
        return JSONResponse(content=response_data)


# VLLM completions
@router.post("/completions", dependencies=[Depends(verify_authorization_header)])
async def completions(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Completions endpoint with optional end-to-end encryption.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request prompt should be encrypted as hex strings
    - Response text will be encrypted as hex strings
    - Only prompt/text is encrypted, not the entire request/response body
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use size-limited read to prevent memory exhaustion attacks
    request_body = await read_body_with_limit(request)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt prompt if encryption is enabled
    if encrypt_enabled and "prompt" in request_json:
        try:
            request_json["prompt"] = decrypt_prompt(request_json["prompt"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt prompt: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt prompt")

    modified_json = strip_empty_tool_calls(request_json)

    # Check if the request is for streaming or non-streaming
    is_stream = modified_json.get(
        "stream", False
    )  # Default to non-streaming if not specified

    modified_request_body = json.dumps(modified_json).encode("utf-8")

    if is_stream:
        # Create a streaming response
        return await stream_vllm_response(
            VLLM_COMPLETIONS_URL,
            request_body,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_client_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
    else:
        # Handle non-streaming response
        response_data = await non_stream_vllm_response(
            VLLM_COMPLETIONS_URL,
            request_body,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_client_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
        return JSONResponse(content=response_data)


# VLLM images generations
@router.post("/images/generations", dependencies=[Depends(verify_authorization_header)])
async def images_generations(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Image generation endpoint with optional end-to-end encryption.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'prompt' field should be encrypted as hex string
    - Response 'data[].b64_json' and 'data[].revised_prompt' fields will be encrypted
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use larger size limit for image requests
    request_body = await read_body_with_limit(request, max_size=MAX_IMAGE_REQUEST_SIZE)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt prompt if encryption is enabled
    if encrypt_enabled and "prompt" in request_json:
        try:
            request_json["prompt"] = _decrypt_field(request_json["prompt"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt prompt: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt prompt")

    # Calculate request hash
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Forward to vLLM images endpoint
    modified_request_body = json.dumps(request_json).encode("utf-8")
    client = get_http_client()
    response = await client.post(VLLM_IMAGES_URL, content=modified_request_body)

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_IMAGES_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response fields if encryption is enabled
    if encrypt_enabled:
        # Type narrowing: encrypt_enabled guarantees these are non-None
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        if "data" in response_data and isinstance(response_data["data"], list):
            for idx, item in enumerate(response_data["data"]):
                log.debug(
                    f"Processing image generation response item {idx}, "
                    f"keys: {list(item.keys())}"
                )
                if "b64_json" in item and item["b64_json"]:
                    b64_value = item["b64_json"]
                    log.debug(
                        f"Encrypting b64_json for item {idx}, "
                        f"type: {type(b64_value).__name__}, "
                        f"length: {len(b64_value) if isinstance(b64_value, (str, bytes)) else 'N/A'}"
                    )
                    item["b64_json"] = encrypt_text(
                        item["b64_json"], x_client_pub_key, x_signing_algo
                    )
                    log.debug(
                        f"Encrypted b64_json for item {idx}, "
                        f"result length: {len(item['b64_json'])}"
                    )
                else:
                    log.debug(
                        f"Skipping b64_json encryption for item {idx}: "
                        f"in_item={'b64_json' in item}, "
                        f"value_truthy={bool(item.get('b64_json'))}"
                    )
                if "revised_prompt" in item and item["revised_prompt"]:
                    item["revised_prompt"] = encrypt_text(
                        item["revised_prompt"], x_client_pub_key, x_signing_algo
                    )
        else:
            log.warning(
                f"Response data structure unexpected: "
                f"has_data={'data' in response_data}, "
                f"data_is_list={isinstance(response_data.get('data'), list)}"
            )
    else:
        log.debug("Encryption not enabled for images/generations response")

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"img-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# VLLM images edits
@router.post("/images/edits", dependencies=[Depends(verify_authorization_header)])
async def images_edits(
    request: Request,
    image: list[UploadFile] = File(..., alias="image[]"),
    prompt: str = Form(...),
    model: str = Form(...),
    n: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    user: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    background: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Image edit endpoint with optional end-to-end encryption.

    Accepts multipart/form-data with one or more images and a prompt.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'prompt' field should be encrypted as hex string
    - Response 'data[].b64_json' and 'data[].revised_prompt' fields will be encrypted

    Parameters:
    - image: The image(s) to edit (required)
    - prompt: Text description of the desired edit (required)
    - model: The model to use (required)
    - background: Set transparency for background - 'transparent', 'opaque', or 'auto' (optional)
    - mask: Additional image indicating where to edit (fully transparent areas) (optional)
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Decrypt prompt if encryption is enabled
    decrypted_prompt = prompt
    if encrypt_enabled:
        try:
            decrypted_prompt = _decrypt_field(prompt, context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt prompt: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt prompt")

    # Calculate request hash from prompt (images are passed through)
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        # Hash the encrypted prompt for consistency with other endpoints
        request_sha256 = sha256(prompt.encode("utf-8")).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Build multipart data for forwarding
    files = []
    for img in image:
        # Read image with streaming size validation
        content = await read_upload_file_with_limit(img, MAX_IMAGE_REQUEST_SIZE)
        files.append(("image[]", (img.filename or "image", content, img.content_type)))

    # Build form data
    data = {
        "prompt": decrypted_prompt,
        "model": model,
    }
    if n is not None:
        data["n"] = str(n)
    if size is not None:
        data["size"] = size
    if response_format is not None:
        data["response_format"] = response_format
    if user is not None:
        data["user"] = user
    if quality is not None:
        data["quality"] = quality
    if background is not None:
        data["background"] = background

    # Add mask file if provided
    if mask is not None:
        # Read mask with streaming size validation
        mask_content = await read_upload_file_with_limit(mask, MAX_IMAGE_REQUEST_SIZE)
        files.append(("mask", (mask.filename or "mask.png", mask_content, mask.content_type)))

    # Forward to vLLM images edits endpoint
    # Use a separate client without Content-Type header for multipart requests
    # The shared client has Content-Type: application/json which breaks multipart
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as multipart_client:
        response = await multipart_client.post(
            VLLM_IMAGES_EDITS_URL,
            files=files,
            data=data,
        )

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_IMAGES_EDITS_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response fields if encryption is enabled
    if encrypt_enabled:
        # Type narrowing: encrypt_enabled guarantees these are non-None
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        if "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if "b64_json" in item and item["b64_json"]:
                    item["b64_json"] = encrypt_text(
                        item["b64_json"], x_client_pub_key, x_signing_algo
                    )
                if "revised_prompt" in item and item["revised_prompt"]:
                    item["revised_prompt"] = encrypt_text(
                        item["revised_prompt"], x_client_pub_key, x_signing_algo
                    )

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"img-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# Audio transcriptions
@router.post("/audio/transcriptions", dependencies=[Depends(verify_authorization_header)])
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    timestamp_granularities: Optional[str] = Form(None),
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Audio transcription endpoint with optional end-to-end encryption.

    Transcribes audio into the input language.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'prompt' field should be encrypted as hex string
    - Response 'text' field will be encrypted

    Parameters:
    - file: The audio file to transcribe (required)
    - model: ID of the model to use (required)
    - language: Language of input audio in ISO-639-1 format (optional)
    - prompt: Optional text to guide the model's style (optional)
    - response_format: Output format: json, text, srt, verbose_json, vtt (optional)
    - temperature: Sampling temperature between 0 and 1 (optional)
    - timestamp_granularities: word, segment, or both (optional)
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Decrypt prompt if encryption is enabled and prompt is provided
    decrypted_prompt = prompt
    if encrypt_enabled and prompt:
        try:
            decrypted_prompt = _decrypt_field(prompt, context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt prompt: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt prompt")

    # Calculate request hash
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        # Hash the prompt if provided, otherwise fall back to model name.
        # Note: Audio file content is NOT included in hash for performance reasons
        # and consistency with /images/edits (which also only hashes the prompt).
        # The model name provides a stable fallback for prompt-less requests.
        hash_content = (prompt or model).encode("utf-8")
        request_sha256 = sha256(hash_content).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Read and validate audio file with streaming size check
    audio_content = await read_upload_file_with_limit(file, MAX_AUDIO_REQUEST_SIZE)

    # Build multipart data for forwarding
    files = [("file", (file.filename or "audio", audio_content, file.content_type))]

    # Build form data
    data = {"model": model}
    if language is not None:
        data["language"] = language
    if decrypted_prompt is not None:
        data["prompt"] = decrypted_prompt
    if response_format is not None:
        data["response_format"] = response_format
    if temperature is not None:
        data["temperature"] = str(temperature)
    if timestamp_granularities is not None:
        data["timestamp_granularities"] = timestamp_granularities

    # Forward to vLLM transcriptions endpoint
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as multipart_client:
        response = await multipart_client.post(
            VLLM_TRANSCRIPTIONS_URL,
            files=files,
            data=data,
        )

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_TRANSCRIPTIONS_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response text if encryption is enabled
    if encrypt_enabled:
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        if "text" in response_data and response_data["text"]:
            response_data["text"] = encrypt_text(
                response_data["text"], x_client_pub_key, x_signing_algo
            )

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"trans-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# VLLM embeddings
@router.post("/embeddings", dependencies=[Depends(verify_authorization_header)])
async def embeddings(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Embeddings endpoint with optional end-to-end encryption.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'input' field should be encrypted as hex string (for string inputs)
    - Response 'data[].embedding' arrays will be encrypted as hex strings
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use size-limited read to prevent memory exhaustion attacks
    request_body = await read_body_with_limit(request)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt input if encryption is enabled
    # Input can be: string, array of strings, array of integers, or array of arrays of integers
    # Only decrypt string inputs, not token IDs (integers)
    if encrypt_enabled and "input" in request_json:
        try:
            request_json["input"] = decrypt_prompt(request_json["input"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt input: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt input")

    # Calculate request hash
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Forward to vLLM embeddings endpoint
    modified_request_body = json.dumps(request_json).encode("utf-8")
    client = get_http_client()
    response = await client.post(VLLM_EMBEDDINGS_URL, content=modified_request_body)

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_EMBEDDINGS_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response embeddings if encryption is enabled
    if encrypt_enabled:
        # Type narrowing: encrypt_enabled guarantees these are non-None
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        if "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if "embedding" in item and item["embedding"]:
                    item["embedding"] = encrypt_embedding(
                        item["embedding"], x_client_pub_key, x_signing_algo
                    )

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"emb-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# VLLM rerank
@router.post("/rerank", dependencies=[Depends(verify_authorization_header)])
async def rerank(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Reranking endpoint with optional end-to-end encryption.

    Reranks documents based on their relevance to a query.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'query' field should be encrypted as hex string
    - Request 'documents' array elements should be encrypted as hex strings
      (supports both string documents and objects with 'text' field)
    - Response 'results[].document.text' fields will be encrypted (if return_documents=true)

    Parameters:
    - model: The model to use for reranking (required)
    - query: The query to rank documents against (required)
    - documents: Array of documents to rerank (required)
      - Can be strings or objects with 'text' field
    - top_n: Number of top results to return (optional)
    - return_documents: Whether to include documents in response (optional, default true)
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use size-limited read to prevent memory exhaustion attacks
    request_body = await read_body_with_limit(request)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt query if encryption is enabled
    if encrypt_enabled and "query" in request_json:
        try:
            request_json["query"] = _decrypt_field(request_json["query"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt query: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt query")

    # Decrypt documents if encryption is enabled
    if encrypt_enabled and "documents" in request_json:
        try:
            request_json["documents"] = decrypt_rerank_documents(
                request_json["documents"], context
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt documents: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt documents")

    # Calculate request hash
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Forward to vLLM rerank endpoint
    modified_request_body = json.dumps(request_json).encode("utf-8")
    client = get_http_client()
    response = await client.post(VLLM_RERANK_URL, content=modified_request_body)

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_RERANK_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response document text if encryption is enabled
    if encrypt_enabled:
        # Type narrowing: encrypt_enabled guarantees these are non-None
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        if "results" in response_data and isinstance(response_data["results"], list):
            for result in response_data["results"]:
                # Handle document.text field (when return_documents=true)
                if "document" in result and isinstance(result["document"], dict):
                    if "text" in result["document"] and result["document"]["text"]:
                        result["document"]["text"] = encrypt_text(
                            result["document"]["text"], x_client_pub_key, x_signing_algo
                        )

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"rerank-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# VLLM score
@router.post("/score", dependencies=[Depends(verify_authorization_header)])
async def score(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
    x_signing_algo: Optional[str] = Header(None, alias="X-Signing-Algo"),
    x_client_pub_key: Optional[str] = Header(None, alias="X-Client-Pub-Key"),
):
    """
    Score endpoint with optional end-to-end encryption.

    Computes a similarity/relevance score between two texts.

    Supports both plain text and encrypted requests/responses.

    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Client-Pub-Key: Client's public key in hex format (required if encryption is enabled)

    When encryption is enabled:
    - Request 'text_1' and 'text_2' fields should be encrypted as hex strings
    - Response 'score' field will be encrypted as hex string

    Parameters:
    - model: The model to use for scoring (required)
    - text_1: First text (e.g., a query) (required)
    - text_2: Second text (e.g., a document) (required)
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_client_pub_key is not None

    # Validate encryption headers and get signing context if encryption is enabled
    context = (
        validate_encryption_headers(x_signing_algo, x_client_pub_key)
        if encrypt_enabled
        else None
    )

    # Use size-limited read to prevent memory exhaustion attacks
    request_body = await read_body_with_limit(request)

    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {str(e)}"
        )

    # Decrypt text_1 if encryption is enabled
    if encrypt_enabled and "text_1" in request_json:
        try:
            request_json["text_1"] = _decrypt_field(request_json["text_1"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt text_1: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt text_1")

    # Decrypt text_2 if encryption is enabled
    if encrypt_enabled and "text_2" in request_json:
        try:
            request_json["text_2"] = _decrypt_field(request_json["text_2"], context)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Failed to decrypt text_2: {type(e).__name__}")
            raise HTTPException(status_code=400, detail="Failed to decrypt text_2")

    # Calculate request hash
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Using client-provided request hash: {request_sha256}")
    else:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Calculated request hash: {request_sha256}")

    # Forward to vLLM score endpoint
    modified_request_body = json.dumps(request_json).encode("utf-8")
    client = get_http_client()
    response = await client.post(VLLM_SCORE_URL, content=modified_request_body)

    if response.status_code != 200:
        error_detail = response.text
        log.error(
            f"Upstream service error from {VLLM_SCORE_URL}: "
            f"{response.status_code} - {error_detail}"
        )
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    response_data = response.json()

    # Encrypt response score if encryption is enabled
    if encrypt_enabled:
        # Type narrowing: encrypt_enabled guarantees these are non-None
        assert x_client_pub_key is not None
        assert x_signing_algo is not None
        # Encrypt the score field (convert to string first if it's a number)
        if "score" in response_data and response_data["score"] is not None:
            score_value = response_data["score"]
            # Convert score to string for encryption
            score_str = json.dumps(score_value)
            response_data["score"] = encrypt_text(
                score_str, x_client_pub_key, x_signing_algo
            )

    # Generate response ID if not present
    response_id = response_data.get("id")
    if not response_id:
        response_id = f"score-{uuid.uuid4().hex[:24]}"
        response_data["id"] = response_id

    # Hash the response and cache signature
    encrypted_response_body = json.dumps(response_data, separators=(",", ":")).encode(
        "utf-8"
    )
    response_sha256 = sha256(encrypted_response_body).hexdigest()
    cache.set_chat(
        response_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
    )

    return JSONResponse(content=response_data)


# VLLM tokenize
@router.post("/tokenize", dependencies=[Depends(verify_authorization_header)])
async def tokenize(request: Request):
    """
    Proxy tokenization requests to vLLM backend.
    Uses size-limited read to prevent memory exhaustion attacks.
    """
    request_body = await read_body_with_limit(request)

    client = get_http_client()
    response = await client.post(
        VLLM_TOKENIZE_URL,
        content=request_body,
        timeout=httpx.Timeout(TIMEOUT_TOKENIZE),
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to tokenize"
        )

    return JSONResponse(content=response.json())


# Get signature for chat_id of chat history
@router.get("/signature/{chat_id}", dependencies=[Depends(verify_authorization_header)])
async def signature(request: Request, chat_id: str, signing_algo: str = None):
    cache_value = cache.get_chat(chat_id)
    if cache_value is None:
        return not_found("Chat id not found or expired")

    signature = None
    signing_algo = ECDSA if signing_algo is None else signing_algo

    # Retrieve the cached request and response
    try:
        value = json.loads(cache_value)
    except Exception as e:
        log.error(
            f"Failed to parse cache value for chat_id={chat_id}: {type(e).__name__}"
        )
        return unexpect_error("Failed to parse the cache value")

    signing_address = None
    if signing_algo == ECDSA:
        signature = value.get("signature_ecdsa")
        signing_address = value.get("signing_address_ecdsa")
    elif signing_algo == ED25519:
        signature = value.get("signature_ed25519")
        signing_address = value.get("signing_address_ed25519")
    else:
        return invalid_signing_algo()

    return dict(
        text=value.get("text"),
        signature=signature,
        signing_address=signing_address,
        signing_algo=signing_algo,
    )


# Metrics of vLLM instance
@router.get("/metrics")
async def metrics(request: Request):
    client = get_http_client()
    response = await client.get(VLLM_METRICS_URL)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch metrics"
        )
    return PlainTextResponse(response.text)


@router.get("/models")
async def models(request: Request):
    client = get_http_client()
    response = await client.get(VLLM_MODELS_URL)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch models"
        )
    return JSONResponse(content=response.json())


async def handle_passthrough(
    request: Request,
    backend_url: str,
    path: str,
    x_request_hash: Optional[str] = None,
):
    """
    Core passthrough logic shared by /v1/ and root-level catch-all routes.

    Forwards requests to the backend vLLM service without parsing or modifying them.

    Signature support:
    - For JSON responses: extracts or injects 'id' field, signs request/response hashes
    - For SSE streaming: extracts 'id' from first chunk or appends final event with generated id
    - Signatures can be retrieved via GET /v1/signature/{id}

    Note: This does NOT support end-to-end encryption.
    For encrypted requests, use the specific endpoint handlers.
    """
    # Prevent path traversal attacks
    if ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Append query string to backend URL
    if request.url.query:
        backend_url = f"{backend_url}?{request.url.query}"

    log.info(f"Passthrough: {request.method} {request.url.path} -> {backend_url}")

    # Read request body if present
    request_body = None
    if request.method in ("POST", "PUT", "PATCH"):
        request_body = await read_body_with_limit(request)

    # Calculate request hash for signature
    if x_request_hash:
        request_sha256 = x_request_hash
        log.info(f"Passthrough: Using client-provided request hash: {request_sha256}")
    elif request_body:
        request_sha256 = sha256(request_body).hexdigest()
        log.debug(f"Passthrough: Calculated request hash: {request_sha256}")
    else:
        # For GET/DELETE/etc without body, hash the path and query
        hash_content = f"{path}?{request.url.query}" if request.url.query else path
        request_sha256 = sha256(hash_content.encode()).hexdigest()
        log.debug(f"Passthrough: Calculated request hash from path: {request_sha256}")

    client = get_http_client()

    # Forward headers (excluding those httpx handles or that can cause issues)
    forward_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ("host", "content-length", "transfer-encoding")
    }

    # Build and send the request, always in streaming mode to inspect headers first
    req = client.build_request(
        method=request.method,
        url=backend_url,
        content=request_body,
        headers=forward_headers,
    )
    response = await client.send(req, stream=True)

    # Do not expose detailed upstream errors
    if response.status_code >= 400:
        await response.aclose()
        raise HTTPException(
            status_code=response.status_code, detail="Upstream service error"
        )

    # Check content type for response handling
    content_type = response.headers.get("content-type", "")
    is_stream_response = "text/event-stream" in content_type
    is_json_response = "application/json" in content_type

    if is_stream_response:
        # Streaming SSE response with signature support
        response_id = None
        h = sha256()

        async def generate_stream_with_signature(resp):
            nonlocal response_id, h
            try:
                async for chunk in resp.aiter_text():
                    # Hash the chunk
                    h.update(chunk.encode())

                    # Try to extract id from first data chunk if not found yet
                    if response_id is None and chunk.strip():
                        # Parse SSE data lines
                        for line in chunk.split("\n"):
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    if "id" in data:
                                        response_id = data["id"]
                                        break
                                except (json.JSONDecodeError, ValueError):
                                    pass

                    yield chunk

                # After stream completes, sign and cache
                response_sha256 = h.hexdigest()

                # Generate id if not found in stream
                final_id = response_id
                if not final_id:
                    final_id = f"passthrough-{uuid.uuid4().hex[:24]}"
                    # Append final SSE event with generated id
                    final_event = f"data: {{\"id\": \"{final_id}\"}}\n\n"
                    # Update hash with final event
                    h.update(final_event.encode())
                    response_sha256 = h.hexdigest()
                    yield final_event

                # Cache signature
                cache.set_chat(
                    final_id,
                    json.dumps(sign_chat(f"{request_sha256}:{response_sha256}")),
                )
                log.debug(f"Passthrough: Cached signature for streaming response id={final_id}")

            finally:
                await resp.aclose()

        return StreamingResponse(
            generate_stream_with_signature(response),
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )

    elif is_json_response:
        # Non-streaming JSON response with signature support
        content = await response.aread()
        await response.aclose()

        try:
            response_data = json.loads(content)

            # Extract or generate id
            response_id = response_data.get("id")
            if not response_id:
                response_id = f"passthrough-{uuid.uuid4().hex[:24]}"
                response_data["id"] = response_id
                log.debug(f"Passthrough: Generated response id={response_id}")

            # Serialize response for hashing (with injected id if needed)
            response_body = json.dumps(response_data, separators=(",", ":")).encode("utf-8")
            response_sha256 = sha256(response_body).hexdigest()

            # Cache signature
            cache.set_chat(
                response_id,
                json.dumps(sign_chat(f"{request_sha256}:{response_sha256}")),
            )
            log.debug(f"Passthrough: Cached signature for JSON response id={response_id}")

            return JSONResponse(
                content=response_data,
                status_code=response.status_code,
            )

        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return as-is without signature
            log.warning("Passthrough: JSON response could not be parsed, returning without signature")
            response_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in ("transfer-encoding", "content-encoding", "content-length")
            }
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
            )

    else:
        # Non-JSON, non-streaming response - return as-is without signature
        content = await response.aread()
        await response.aclose()

        # Preserve response headers, excluding those that can cause issues
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in ("transfer-encoding", "content-encoding", "content-length")
        }

        return Response(
            content=content,
            status_code=response.status_code,
            headers=response_headers,
        )


# Catch-all passthrough for undefined /v1/ endpoints
# This must be registered LAST to avoid catching defined routes
@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    dependencies=[Depends(verify_authorization_header)],
)
async def passthrough(
    request: Request,
    path: str,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
):
    backend_url = f"{VLLM_BASE_URL}/v1/{path}"
    return await handle_passthrough(request, backend_url, path, x_request_hash)
