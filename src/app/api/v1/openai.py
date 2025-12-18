import json
import os
from hashlib import sha256
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Header, Query
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    Response,
)

from app.api.helper.auth import verify_authorization_header
from app.api.response.response import (
    invalid_signing_algo,
    not_found,
    unexpect_error,
)
from app.cache.cache import cache
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
from app.encryption.encryption import (
    encrypt_data,
    decrypt_data,
)

router = APIRouter(tags=["openai"])

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000")
VLLM_URL = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/completions"
VLLM_TOKENIZE_URL = f"{VLLM_BASE_URL}/tokenize"
VLLM_METRICS_URL = f"{VLLM_BASE_URL}/metrics"
VLLM_MODELS_URL = f"{VLLM_BASE_URL}/v1/models"
TIMEOUT = 60 * 10
TIMEOUT_TOKENIZE = 10

COMMON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


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


def _decrypt_field(field_value: str, context: SigningContext) -> str:
    """
    Decrypt a field value if it's encrypted (hex string).
    Returns decrypted value or original value if not encrypted.
    """
    if not isinstance(field_value, str) or len(field_value) == 0:
        return field_value

    # Check if it's a valid hex string (even length, hex characters only)
    # Encrypted data is typically longer, so we check for minimum length
    if len(field_value) >= 64 and len(field_value) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in field_value):
        try:
            # Try to decode as hex and decrypt
            encrypted_data = bytes.fromhex(field_value)
            decrypted_content = decrypt_data(encrypted_data, context)
            return decrypted_content.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            # Not valid hex or decryption failed, treat as plain text
            pass
        except Exception as e:
            log.error(f"Failed to decrypt field: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decrypt field: {str(e)}"
            )

    return field_value


def decrypt_message_content(message: dict, context: SigningContext) -> dict:
    """
    Decrypt the content and reasoning_content fields of a message if they're encrypted.
    Expected format: {"content": "hex_string"} (encrypted) or {"content": "plain_text"} (unencrypted)
    If content/reasoning_content is a valid hex string, it will be treated as encrypted and decrypted.
    """
    message = dict(message)  # Create a copy to avoid mutating the original
    
    # Decrypt content field if present
    if "content" in message and message["content"] is not None and message["content"]:
        message["content"] = _decrypt_field(message["content"], context)

    return message


def encrypt_message_content(message: dict, client_public_key: str, signing_algo: str) -> dict:
    """
    Encrypt the content and reasoning_content fields of a message.
    Returns message with content/reasoning_content as hex strings (plain string format matching chat completions)
    """
    message = dict(message)  # Create a copy to avoid mutating the original

    # Encrypt content field if present
    if "content" in message and message["content"] is not None and message["content"]:
        content = message["content"]
        if isinstance(content, str):
            try:
                encrypted_data = encrypt_data(content.encode("utf-8"), client_public_key, signing_algo)
                message["content"] = encrypted_data.hex()
            except Exception as e:
                log.error(f"Failed to encrypt message content: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to encrypt message content: {str(e)}"
                )

    # Encrypt reasoning_content field if present
    if "reasoning_content" in message and message["reasoning_content"] is not None and message["reasoning_content"]:
        reasoning_content = message["reasoning_content"]
        if isinstance(reasoning_content, str):
            try:
                encrypted_data = encrypt_data(reasoning_content.encode("utf-8"), client_public_key, signing_algo)
                message["reasoning_content"] = encrypted_data.hex()
            except Exception as e:
                log.error(f"Failed to encrypt message reasoning_content: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to encrypt message reasoning_content: {str(e)}"
                )

    return message


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
                    error_message = f"Failed to parse the first chunk: {type(e).__name__}: {e}"
                    log.error(error_message)
                    raise Exception(error_message)

            # Hash the plain chunk first (for attestation, we hash what the model processes)
            h.update(chunk.encode())

            # Encrypt only message content if needed
            if encrypt_response and client_public_key and signing_algo:
                # Skip encryption for empty or done chunks
                if not data or data == "[DONE]":
                    yield chunk
                    continue

                try:
                    chunk_data = json.loads(data)

                    # Encrypt content and reasoning_content in choices[].delta or choices[].message
                    if "choices" in chunk_data:
                        for choice in chunk_data["choices"]:
                            # Handle delta fields
                            if "delta" in choice:
                                choice["delta"] = encrypt_message_content(
                                    choice["delta"], client_public_key, signing_algo
                                )

                    # Create the modified chunk string
                    modified_chunk = f"data: {json.dumps(chunk_data)}\n\n"
                    # Yield the encrypted chunk
                    yield modified_chunk
                except Exception as e:
                    log.error(f"Failed to encrypt chunk content: {e}")
                    # Yield error chunk
                    error_chunk = f'data: {{"error": "Encryption failed: {str(e)}"}}\n\n'
                    yield error_chunk
            else:
                yield chunk

        response_sha256 = h.hexdigest()
        # Cache the full request and response using the extracted cache key
        if chat_id:
            cache.set_chat(
                chat_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
            )
        else:
            error_message = "Chat id could not be extracted from the response"
            log.error(error_message)
            raise Exception(error_message)

    client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT), headers=COMMON_HEADERS)
    # Forward the request to the vllm backend
    req = client.build_request("POST", url, content=modified_request_body)
    response = await client.send(req, stream=True)
    # If not 200, return the error response directly without streaming
    if response.status_code != 200:
        error_content = await response.aread()
        await response.aclose()
        await client.aclose()

        return Response(
            content=error_content,
            status_code=response.status_code,
            headers=response.headers,
        )

    return StreamingResponse(
        generate_stream(response),
        background=BackgroundTasks([response.aclose, client.aclose]),
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

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT), headers=COMMON_HEADERS
    ) as client:
        response = await client.post(url, content=modified_request_body)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Upstream service error")

        response_data = response.json()

        # Encrypt message content
        if encrypt_response and client_public_key and signing_algo:
            if "choices" in response_data:
                for choice in response_data["choices"]:
                    if "message" in choice:
                        choice["message"] = encrypt_message_content(
                            choice["message"], client_public_key, signing_algo
                        )

        # Cache the request-response pair using the chat ID
        chat_id = response_data.get("id")
        if chat_id:
            response_sha256 = sha256(response.content).hexdigest()
            cache.set_chat(
                chat_id, json.dumps(sign_chat(f"{request_sha256}:{response_sha256}"))
            )
        else:
            raise Exception("Chat id could not be extracted from the response")

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
        raise HTTPException(status_code=404, detail="Signing address not found on this server")
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
    x_signing_pub_key: Optional[str] = Header(None, alias="X-Signing-Pub-Key"),
):
    """
    Chat completions endpoint with optional end-to-end encryption.
    
    Supports both plain text and encrypted requests/responses.
    
    Optional encryption headers (both must be provided to enable encryption):
    - X-Signing-Algo: Either 'ecdsa' or 'ed25519' (required if encryption is enabled)
    - X-Signing-Pub-Key: Client's public key in hex format (required if encryption is enabled)
    
    When encryption is enabled:
    - Request message content should be encrypted as hex strings
    - Response message content will be encrypted as hex strings
    - Only message content is encrypted, not the entire request/response body
    """
    # Check if encryption is requested
    encrypt_enabled = x_signing_algo is not None and x_signing_pub_key is not None
    
    if encrypt_enabled:
        # Validate signing algorithm
        if x_signing_algo not in [ECDSA, ED25519]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid X-Signing-Algo. Must be '{ECDSA}' or '{ED25519}'"
            )
        
        # Get the signing context for decryption
        context = ecdsa_context if x_signing_algo == ECDSA else ed25519_context
    
    # Get the request body
    request_body = await request.body()
    
    # Parse the request JSON
    try:
        request_json = json.loads(request_body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in request body: {str(e)}"
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
                log.error(f"Failed to decrypt message content: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decrypt message content: {str(e)}"
                )
        request_json["messages"] = decrypted_messages
    
    modified_json = strip_empty_tool_calls(request_json)

    # Check if the request is for streaming or non-streaming
    is_stream = modified_json.get(
        "stream", False
    )  # Default to non-streaming if not specified

    modified_request_body = json.dumps(modified_json).encode("utf-8")
    
    # Use decrypted body for hash calculation if encryption is enabled, otherwise use original
    body_for_hash = modified_request_body if encrypt_enabled else request_body
    
    if is_stream:
        # Create a streaming response
        return await stream_vllm_response(
            VLLM_URL,
            body_for_hash,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_signing_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
    else:
        # Handle non-streaming response
        response_data = await non_stream_vllm_response(
            VLLM_URL,
            body_for_hash,
            modified_request_body,
            x_request_hash,
            encrypt_response=encrypt_enabled,
            client_public_key=x_signing_pub_key if encrypt_enabled else None,
            signing_algo=x_signing_algo if encrypt_enabled else None,
        )
        return JSONResponse(content=response_data)


# VLLM completions
@router.post("/completions", dependencies=[Depends(verify_authorization_header)])
async def completions(
    request: Request,
    x_request_hash: Optional[str] = Header(None, alias="X-Request-Hash"),
):
    # Keep original request body to calculate the request hash for attestation
    request_body = await request.body()
    request_json = json.loads(request_body)
    modified_json = strip_empty_tool_calls(request_json)

    # Check if the request is for streaming or non-streaming
    is_stream = modified_json.get(
        "stream", False
    )  # Default to non-streaming if not specified

    modified_request_body = json.dumps(modified_json).encode("utf-8")
    if is_stream:
        # Create a streaming response
        return await stream_vllm_response(
            VLLM_COMPLETIONS_URL, request_body, modified_request_body, x_request_hash
        )
    else:
        # Handle non-streaming response
        response_data = await non_stream_vllm_response(
            VLLM_COMPLETIONS_URL, request_body, modified_request_body, x_request_hash
        )
        return JSONResponse(content=response_data)


# VLLM tokenize
@router.post("/tokenize", dependencies=[Depends(verify_authorization_header)])
async def tokenize(request: Request):
    """
    Proxy tokenization requests to vLLM backend.
    """
    request_body = await request.body()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT_TOKENIZE), headers=COMMON_HEADERS
    ) as client:
        response = await client.post(
            VLLM_TOKENIZE_URL,
            content=request_body
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to tokenize"
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
        log.error(f"Failed to parse cache value for chat_id={chat_id}: {type(e).__name__}")
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        response = await client.get(VLLM_METRICS_URL)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch metrics")
        return PlainTextResponse(response.text)


@router.get("/models")
async def models(request: Request):
    async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT)) as client:
        response = await client.get(VLLM_MODELS_URL)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch models")
        return JSONResponse(content=response.json())
