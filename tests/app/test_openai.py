from unittest.mock import patch, AsyncMock
import httpx
import pytest
from fastapi.testclient import TestClient
import json

# Import and setup test environment before importing app
from tests.app.test_helpers import setup_test_environment, TEST_AUTH_HEADER

# Setup all mocks before importing app
setup_test_environment()

# Replace the quote module with our mock before importing app
import sys

sys.modules["app.quote.quote"] = __import__("tests.app.mock_quote", fromlist=[""])

# Now we can safely import app code
from app.main import app
from app.api.v1.openai import (
    VLLM_URL,
    VLLM_BASE_URL,
    VLLM_EMBEDDINGS_URL,
    VLLM_IMAGES_EDITS_URL,
    VLLM_TRANSCRIPTIONS_URL,
    VLLM_RERANK_URL,
    VLLM_SCORE_URL,
)
from tests.app.mock_quote import ED25519, ECDSA, ecdsa_quote, ed25519_quote

client = TestClient(app)


async def yield_sse_response(data_list):
    for data in data_list:
        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_stream_chat_completions_success(respx_mock):
    # Test request data
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Mock streaming response data
    chat_id = "chatcmpl-123"
    responses = [
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        },
    ]

    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Make request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify response
    assert response.status_code == 200
    assert route.called

    # Collect all streaming responses
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            chunk = json.loads(line.replace("data: ", ""))
            chunks.append(chunk)

    # Verify streaming response content
    assert len(chunks) == 3
    assert chunks[0]["id"] == chat_id
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_stream_chat_completions_upstream_error(respx_mock):
    # Test request data
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Setup RESPX mock with a 400 error response
    error_response = {
        "error": {
            "message": "Invalid request parameters",
            "type": "invalid_request_error",
            "code": 400,
        }
    }
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(400, json=error_response)
    )

    # Make request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify response
    assert response.status_code == 400
    assert route.called

    # Verify error response content
    response_data = response.json()
    assert "error" in response_data
    assert response_data["error"]["message"] == "Invalid request parameters"
    assert response_data["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_signature_default_algo():
    # Setup test data
    chat_id = "test-chat-123"
    test_data = "test request:response data"

    # Create properly formatted cache data
    cache_data = json.dumps(
        {
            "text": test_data,
            "signature_ecdsa": ecdsa_quote.sign(test_data),
            "signing_address_ecdsa": ecdsa_quote.signing_address,
            "signature_ed25519": ed25519_quote.sign(test_data),
            "signing_address_ed25519": ed25519_quote.signing_address,
        }
    )

    # Only mock the cache, use real quote object
    with patch("app.api.v1.openai.cache") as mock_cache:
        # Setup mock cache
        mock_cache.get_chat.return_value = cache_data

        # Make request
        response = client.get(
            f"/v1/signature/{chat_id}", headers={"Authorization": TEST_AUTH_HEADER}
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["text"] == test_data
        assert len(response_data["signature"]) > 0  # Real signature will have content
        assert response_data["signing_algo"] == ECDSA


@pytest.mark.asyncio
async def test_signature_explicit_algo():
    # Setup test data
    chat_id = "test-chat-123"
    test_data = "test request:response data"

    # Create properly formatted cache data
    cache_data = json.dumps(
        {
            "text": test_data,
            "signature_ecdsa": ecdsa_quote.sign(test_data),
            "signing_address_ecdsa": ecdsa_quote.signing_address,
            "signature_ed25519": ed25519_quote.sign(test_data),
            "signing_address_ed25519": ed25519_quote.signing_address,
        }
    )

    # Only mock the cache, use real quote object
    with patch("app.api.v1.openai.cache") as mock_cache:
        # Setup mock cache
        mock_cache.get_chat.return_value = cache_data

        # Make request with explicit algorithm
        explicit_algo = ED25519  # Use ED25519 explicitly
        response = client.get(
            f"/v1/signature/{chat_id}?signing_algo={explicit_algo}",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["text"] == test_data
        assert len(response_data["signature"]) > 0  # Real signature will have content
        assert response_data["signing_algo"] == explicit_algo


@pytest.mark.asyncio
async def test_signature_invalid_algo():
    chat_id = "test-chat-123"

    # Create properly formatted cache data
    cache_data = json.dumps(
        {
            "text": "test data",
            "signature_ecdsa": "test_sig",
            "signing_address_ecdsa": "test_addr",
            "signature_ed25519": "test_sig",
            "signing_address_ed25519": "test_addr",
        }
    )

    # Only mock the cache
    with patch("app.api.v1.openai.cache") as mock_cache:
        mock_cache.get_chat.return_value = cache_data

        # Make request with invalid algorithm
        response = client.get(
            f"/v1/signature/{chat_id}?signing_algo=invalid-algo",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify error response
        assert response.status_code == 400
        response_data = response.json()
        assert (
            response_data["error"]["message"]
            == "Invalid signing algorithm. Must be 'ed25519' or 'ecdsa'"
        )
        assert response_data["error"]["type"] == "invalid_signing_algo"


@pytest.mark.asyncio
async def test_signature_chat_not_found():
    chat_id = "nonexistent-chat"

    # Mock the cache to return None for chat not found
    with patch("app.api.v1.openai.cache") as mock_cache:
        mock_cache.get_chat.return_value = None

        # Make request
        response = client.get(
            f"/v1/signature/{chat_id}", headers={"Authorization": TEST_AUTH_HEADER}
        )

        # Verify error response
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["error"]["message"] == "Chat id not found or expired"
        assert response_data["error"]["type"] == "not_found"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_request_hash_streaming(respx_mock):
    # Test request data
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    # Pre-calculated hash for the request
    request_body = json.dumps(request_data).encode("utf-8")
    expected_hash = "custom-hash-from-client"

    # Mock streaming response data
    chat_id = "chatcmpl-123"
    responses = [
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        },
    ]

    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Mock cache and logging to verify hash usage
    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:

        # Make request with X-Request-Hash header
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        # Verify cache was called with the custom hash
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_request_hash_non_streaming(respx_mock):
    # Test request data
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Pre-calculated hash for the request
    expected_hash = "custom-hash-from-client"

    # Mock non-streaming response data
    chat_id = "chatcmpl-456"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello back"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Mock cache and logging to verify hash usage
    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:

        # Make request with X-Request-Hash header
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        # Verify cache was called with the custom hash
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_completions_with_request_hash_streaming(respx_mock):
    # Test request data
    request_data = {"model": "test-model", "prompt": "Hello", "stream": True}

    # Pre-calculated hash for the request
    expected_hash = "custom-completions-hash"

    # Mock streaming response data
    completion_id = "cmpl-123"
    responses = [
        {
            "id": completion_id,
            "object": "text_completion",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"text": "Hello", "index": 0, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "text_completion",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"text": " back", "index": 0, "finish_reason": None}],
        },
        {
            "id": completion_id,
            "object": "text_completion",
            "created": 1677825464,
            "model": "test-model",
            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
        },
    ]

    # Setup RESPX mock for completions endpoint
    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Mock cache and logging to verify hash usage
    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:

        # Make request with X-Request-Hash header
        response = client.post(
            "/v1/completions",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        # Verify cache was called with the custom hash
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_completions_with_request_hash_non_streaming(respx_mock):
    # Test request data
    request_data = {"model": "test-model", "prompt": "Hello", "stream": False}

    # Pre-calculated hash for the request
    expected_hash = "custom-completions-hash"

    # Mock non-streaming response data
    completion_id = "cmpl-456"
    response_data = {
        "id": completion_id,
        "object": "text_completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [{"text": "Hello back", "index": 0, "finish_reason": "stop"}],
    }

    # Setup RESPX mock for completions endpoint
    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Mock cache and logging to verify hash usage
    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:

        # Make request with X-Request-Hash header
        response = client.post(
            "/v1/completions",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        # Verify cache was called with the custom hash
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_without_request_hash(respx_mock):
    # Test request data without X-Request-Hash header
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Mock non-streaming response data
    chat_id = "chatcmpl-789"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello back"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Mock cache and logging to verify hash calculation
    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:

        # Make request without X-Request-Hash header
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that hash was calculated (debug log should be called)
        mock_log.debug.assert_called()
        debug_call_args = mock_log.debug.call_args[0][0]
        assert "Calculated request hash:" in debug_call_args

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


# ============================================================================
# Request Size Limiting Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_within_size_limit(respx_mock):
    """Test that requests within size limit succeed normally."""
    # Test request data (small payload - well under 10MB default limit)
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Mock response
    chat_id = "chatcmpl-size-test"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello back"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    assert route.called


@pytest.mark.asyncio
async def test_read_body_with_limit_function_rejects_large_body():
    """Test that read_body_with_limit rejects bodies exceeding the limit."""
    from app.api.v1.openai import read_body_with_limit
    from fastapi import HTTPException

    # Create a mock request with a large body
    class MockRequest:
        def __init__(self, body_content: bytes, content_length: str = None):
            self._body = body_content
            self._headers = {}
            if content_length:
                self._headers["content-length"] = content_length

        @property
        def headers(self):
            return self._headers

        async def stream(self):
            # Yield in chunks
            chunk_size = 100
            for i in range(0, len(self._body), chunk_size):
                yield self._body[i : i + chunk_size]

    # Test with body exceeding limit
    large_body = b"A" * 500
    mock_request = MockRequest(large_body)

    with pytest.raises(HTTPException) as exc_info:
        await read_body_with_limit(mock_request, max_size=100)

    assert exc_info.value.status_code == 413
    assert "Request body too large" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_read_body_with_limit_accepts_valid_body():
    """Test that read_body_with_limit accepts bodies within the limit."""
    from app.api.v1.openai import read_body_with_limit

    class MockRequest:
        def __init__(self, body_content: bytes, content_length: str = None):
            self._body = body_content
            self._headers = {}
            if content_length:
                self._headers["content-length"] = content_length

        @property
        def headers(self):
            return self._headers

        async def stream(self):
            yield self._body

    # Test with body within limit
    small_body = b'{"model": "test"}'
    mock_request = MockRequest(small_body)

    result = await read_body_with_limit(mock_request, max_size=1000)
    assert result == small_body


@pytest.mark.asyncio
async def test_read_body_with_limit_rejects_large_content_length():
    """Test that read_body_with_limit rejects based on Content-Length header."""
    from app.api.v1.openai import read_body_with_limit
    from fastapi import HTTPException

    class MockRequest:
        def __init__(self, body_content: bytes, content_length: str = None):
            self._body = body_content
            self._headers = {}
            if content_length:
                self._headers["content-length"] = content_length

        @property
        def headers(self):
            return self._headers

        async def stream(self):
            yield self._body

    # Test with Content-Length header exceeding limit
    small_body = b'{"model": "test"}'
    mock_request = MockRequest(small_body, content_length="999999999")

    with pytest.raises(HTTPException) as exc_info:
        await read_body_with_limit(mock_request, max_size=1000)

    assert exc_info.value.status_code == 413
    assert "Request body too large" in str(exc_info.value.detail)


# ============================================================================
# Integration Tests for Request Size Limiting via API Endpoints
# ============================================================================


def _create_limited_read_body(original_func, max_size: int):
    """Create a wrapper that calls read_body_with_limit with a custom max_size.
    
    The wrapper accepts max_size kwarg to match the real function signature,
    but ignores it and uses the test-configured max_size instead.
    """
    test_limit = max_size  # Capture the test-configured limit

    async def limited_read_body(request, max_size: int = None):
        # Always use the test-configured limit, ignore caller's max_size
        return await original_func(request, max_size=test_limit)

    return limited_read_body


@pytest.mark.asyncio
async def test_chat_completions_rejects_oversized_request():
    """
    Integration test: Verify that /v1/chat/completions rejects requests
    exceeding the size limit with proper 413 status code and error format.
    """
    from app.api.v1.openai import read_body_with_limit

    # Create an oversized request by including a very large message content
    # We patch read_body_with_limit with a wrapper that uses a small limit
    large_content = "x" * 1000  # 1000 character message
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": large_content}],
        "stream": False,
    }

    # Patch with a wrapper that enforces a 100-byte limit
    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    # Verify 413 Payload Too Large status code
    assert response.status_code == 413

    # Verify error response format matches OpenAI-style error structure
    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data["error"]
    assert "type" in response_data["error"]
    assert response_data["error"]["type"] == "http_exception"

    # Verify the actual error detail is preserved (not replaced with generic message)
    # This is important for 413 errors where the message includes useful info like max size
    assert "Request body too large" in response_data["error"]["message"]
    assert "100 bytes" in response_data["error"]["message"]

    # Verify request_id is included for debugging/support purposes
    assert "request_id" in response_data["error"]
    assert response_data["error"]["request_id"] is not None

    # Verify X-Request-ID header is present
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_completions_rejects_oversized_request():
    """
    Integration test: Verify that /v1/completions rejects requests
    exceeding the size limit with proper 413 status code and error format.
    """
    from app.api.v1.openai import read_body_with_limit

    # Create an oversized request with a large prompt
    large_prompt = "y" * 1000  # 1000 character prompt
    request_data = {
        "model": "test-model",
        "prompt": large_prompt,
        "stream": False,
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    # Verify 413 Payload Too Large status code
    assert response.status_code == 413

    # Verify error response format matches OpenAI-style error structure
    response_data = response.json()
    assert "error" in response_data
    assert "message" in response_data["error"]
    assert "type" in response_data["error"]
    assert response_data["error"]["type"] == "http_exception"

    # Verify the actual error detail is preserved for 413
    assert "Request body too large" in response_data["error"]["message"]

    # Verify request_id is included
    assert "request_id" in response_data["error"]


@pytest.mark.asyncio
async def test_chat_completions_streaming_rejects_oversized_request():
    """
    Integration test: Verify that /v1/chat/completions with streaming enabled
    also rejects oversized requests before attempting to stream.
    """
    from app.api.v1.openai import read_body_with_limit

    large_content = "z" * 1000
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": large_content}],
        "stream": True,  # Streaming enabled
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    # Size check happens before streaming starts, so we get 413
    assert response.status_code == 413
    response_data = response.json()
    assert "error" in response_data
    assert response_data["error"]["type"] == "http_exception"
    assert "Request body too large" in response_data["error"]["message"]
    assert "request_id" in response_data["error"]


@pytest.mark.asyncio
async def test_tokenize_rejects_oversized_request():
    """
    Integration test: Verify that /v1/tokenize endpoint also enforces
    request size limits.
    """
    from app.api.v1.openai import read_body_with_limit

    large_text = "w" * 1000
    request_data = {
        "model": "test-model",
        "text": large_text,
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/tokenize",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 413
    response_data = response.json()
    assert "error" in response_data
    assert response_data["error"]["type"] == "http_exception"
    assert "Request body too large" in response_data["error"]["message"]
    assert "request_id" in response_data["error"]


# ============================================================================
# Request ID Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_request_id_header_returned_on_success(respx_mock):
    """Test that X-Request-ID header is returned on successful requests."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response_data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    assert route.called
    # Verify X-Request-ID header is present
    assert "X-Request-ID" in response.headers
    # Request ID should be a valid UUID format
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) > 0


@pytest.mark.asyncio
@pytest.mark.respx
async def test_request_id_header_preserved_from_client(respx_mock):
    """Test that X-Request-ID from client is preserved in response."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response_data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    client_request_id = "client-provided-request-id-12345"

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-ID": client_request_id,
            },
        )

    assert response.status_code == 200
    assert route.called
    # Verify client's request ID is preserved
    assert response.headers["X-Request-ID"] == client_request_id


# ============================================================================
# Embeddings Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_embeddings_success(respx_mock):
    """Test successful embeddings request."""
    # Test request data
    request_data = {
        "input": "The food was delicious and the waiter...",
        "model": "text-embedding-ada-002",
        "encoding_format": "float",
    }

    # Mock response data
    response_id = "emb-123"
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.0023064255, -0.009327292, 0.015797734],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["embedding"] == [0.0023064255, -0.009327292, 0.015797734]
        assert result["id"] == response_id

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_embeddings_with_array_input(respx_mock):
    """Test embeddings request with array of strings input."""
    # Test request data with array input
    request_data = {
        "input": ["First text", "Second text"],
        "model": "text-embedding-ada-002",
    }

    # Mock response data with multiple embeddings
    response_id = "emb-456"
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.001, 0.002, 0.003],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [0.004, 0.005, 0.006],
                "index": 1,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert len(result["data"]) == 2
        assert result["data"][0]["index"] == 0
        assert result["data"][1]["index"] == 1


@pytest.mark.asyncio
@pytest.mark.respx
async def test_embeddings_upstream_error(respx_mock):
    """Test embeddings request when upstream returns error."""
    request_data = {
        "input": "Test input",
        "model": "text-embedding-ada-002",
    }

    # Setup RESPX mock with error
    error_response = {"error": {"message": "Model not found", "type": "invalid_request_error"}}
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(404, json=error_response)
    )

    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify error response
    assert response.status_code == 404
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_embeddings_with_request_hash(respx_mock):
    """Test embeddings request with X-Request-Hash header."""
    request_data = {
        "input": "Test input",
        "model": "text-embedding-ada-002",
    }

    expected_hash = "custom-embeddings-hash"

    response_id = "emb-789"
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
        "id": response_id,
    }

    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_embeddings_generates_id_if_missing(respx_mock):
    """Test that embeddings endpoint generates ID if not in response."""
    request_data = {
        "input": "Test input",
        "model": "text-embedding-ada-002",
    }

    # Response without ID
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }

    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("emb-")
        assert len(result["id"]) == 28  # "emb-" + 24 hex chars


@pytest.mark.asyncio
async def test_embeddings_rejects_oversized_request():
    """Test that embeddings endpoint rejects oversized requests."""
    from app.api.v1.openai import read_body_with_limit

    large_input = "x" * 1000
    request_data = {
        "input": large_input,
        "model": "text-embedding-ada-002",
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 413
    response_data = response.json()
    assert "detail" in response_data
    assert "Request body too large" in response_data["detail"]


# ============================================================================
# Images Edits Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_success(respx_mock):
    """Test successful image edit request."""
    # Mock response data
    response_id = "img-edit-123"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "revised_prompt": "A lovely gift basket with items",
            }
        ],
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Create test image data
    test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Minimal PNG header + padding

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Create a lovely gift basket",
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", test_image, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert len(result["data"]) == 1
        assert "b64_json" in result["data"][0]
        assert result["id"] == response_id

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_with_multiple_images(respx_mock):
    """Test image edit request with multiple images."""
    response_id = "img-edit-456"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "base64encodedimage",
                "revised_prompt": "Combined images",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_image1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    test_image2 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    test_image3 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Combine these images",
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("image1.png", test_image1, "image/png")),
                ("image[]", ("image2.png", test_image2, "image/png")),
                ("image[]", ("image3.png", test_image3, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_upstream_error(respx_mock):
    """Test image edit request when upstream returns error."""
    error_response = {"error": {"message": "Invalid image", "type": "invalid_request_error"}}
    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(400, json=error_response)
    )

    test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "Edit this image",
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", test_image, "image/png")),
        ],
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_with_request_hash(respx_mock):
    """Test image edit request with X-Request-Hash header."""
    expected_hash = "custom-image-edit-hash"
    response_id = "img-edit-789"
    response_data = {
        "created": 1677825464,
        "data": [{"b64_json": "base64data"}],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Edit this image",
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", test_image, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_generates_id_if_missing(respx_mock):
    """Test that images edits endpoint generates ID if not in response."""
    response_data = {
        "created": 1677825464,
        "data": [{"b64_json": "base64data"}],
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Edit this image",
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", test_image, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("img-")
        assert len(result["id"]) == 28  # "img-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_edits_with_optional_params(respx_mock):
    """Test image edit request with optional parameters."""
    response_id = "img-edit-opts"
    response_data = {
        "created": 1677825464,
        "data": [{"b64_json": "base64data"}],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Edit this image",
                "model": "gpt-image-1.5",
                "n": "2",
                "size": "1024x1024",
                "response_format": "b64_json",
                "quality": "hd",
            },
            files=[
                ("image[]", ("test.png", test_image, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


# Audio transcriptions tests


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_success(respx_mock):
    """Test successful audio transcription request."""
    response_id = "trans-123"
    response_data = {
        "text": "Hello, this is a test transcription.",
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Create test audio data (minimal WAV header + padding)
    test_audio = b"RIFF" + b"\x00" * 100

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
            },
            files=[
                ("file", ("test.wav", test_audio, "audio/wav")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert result["text"] == "Hello, this is a test transcription."
        assert result["id"] == response_id

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_with_optional_params(respx_mock):
    """Test audio transcription request with optional parameters."""
    response_id = "trans-456"
    response_data = {
        "text": "Bonjour, ceci est un test.",
        "id": response_id,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_audio = b"RIFF" + b"\x00" * 100

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "language": "fr",
                "prompt": "This is a French audio clip",
                "response_format": "json",
                "temperature": "0.2",
            },
            files=[
                ("file", ("test.mp3", test_audio, "audio/mpeg")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert result["text"] == "Bonjour, ceci est un test."


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_upstream_error(respx_mock):
    """Test audio transcription request when upstream returns error."""
    error_response = {"error": {"message": "Invalid audio format", "type": "invalid_request_error"}}
    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(400, json=error_response)
    )

    test_audio = b"RIFF" + b"\x00" * 50

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
        },
        files=[
            ("file", ("test.wav", test_audio, "audio/wav")),
        ],
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_with_request_hash(respx_mock):
    """Test audio transcription request with X-Request-Hash header."""
    expected_hash = "custom-transcription-hash"
    response_id = "trans-789"
    response_data = {
        "text": "Test transcription",
        "id": response_id,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_audio = b"RIFF" + b"\x00" * 50

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
            },
            files=[
                ("file", ("test.wav", test_audio, "audio/wav")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_generates_id_if_missing(respx_mock):
    """Test that audio transcriptions endpoint generates ID if not in response."""
    response_data = {
        "text": "Generated ID test",
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_audio = b"RIFF" + b"\x00" * 50

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
            },
            files=[
                ("file", ("test.wav", test_audio, "audio/wav")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("trans-")
        assert len(result["id"]) == 30  # "trans-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_verbose_json_format(respx_mock):
    """Test audio transcription with verbose_json response format."""
    response_id = "trans-verbose"
    response_data = {
        "text": "Hello world",
        "id": response_id,
        "task": "transcribe",
        "language": "english",
        "duration": 2.5,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
            }
        ],
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    test_audio = b"RIFF" + b"\x00" * 100

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
            },
            files=[
                ("file", ("test.wav", test_audio, "audio/wav")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert result["text"] == "Hello world"
        assert result["language"] == "english"
        assert "segments" in result


# ============================================================================
# read_upload_file_with_limit Tests
# ============================================================================


class MockUploadFile:
    """Mock UploadFile for testing read_upload_file_with_limit."""

    def __init__(self, content: bytes, size: int = None, filename: str = "test.bin"):
        self._content = content
        self._position = 0
        self.size = size  # Simulates Content-Length header
        self.filename = filename
        self.content_type = "application/octet-stream"

    async def read(self, size: int = -1) -> bytes:
        if size == -1:
            result = self._content[self._position:]
            self._position = len(self._content)
        else:
            result = self._content[self._position:self._position + size]
            self._position += len(result)
        return result


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_accepts_valid_file():
    """Test that read_upload_file_with_limit accepts files within the limit."""
    from app.api.v1.openai import read_upload_file_with_limit

    small_content = b"Hello, this is a small file."
    mock_file = MockUploadFile(small_content, size=len(small_content))

    result = await read_upload_file_with_limit(mock_file, max_size=1000)
    assert result == small_content


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_rejects_large_file_by_size_attribute():
    """Test that read_upload_file_with_limit rejects based on file.size attribute (early check)."""
    from app.api.v1.openai import read_upload_file_with_limit
    from fastapi import HTTPException

    # File with size attribute exceeding limit - should be rejected before reading
    large_content = b"X" * 100
    mock_file = MockUploadFile(large_content, size=999999)  # Size attr says it's huge

    with pytest.raises(HTTPException) as exc_info:
        await read_upload_file_with_limit(mock_file, max_size=1000)

    assert exc_info.value.status_code == 413
    assert "File too large" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_rejects_large_file_during_streaming():
    """Test that read_upload_file_with_limit rejects files exceeding limit during streaming."""
    from app.api.v1.openai import read_upload_file_with_limit
    from fastapi import HTTPException

    # File without size attribute - must be validated during streaming
    large_content = b"Y" * 5000  # 5KB content
    mock_file = MockUploadFile(large_content, size=None)  # No size attribute

    with pytest.raises(HTTPException) as exc_info:
        await read_upload_file_with_limit(mock_file, max_size=1000)

    assert exc_info.value.status_code == 413
    assert "File too large" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_handles_empty_file():
    """Test that read_upload_file_with_limit handles empty files correctly."""
    from app.api.v1.openai import read_upload_file_with_limit

    empty_content = b""
    mock_file = MockUploadFile(empty_content, size=0)

    result = await read_upload_file_with_limit(mock_file, max_size=1000)
    assert result == b""


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_exact_limit():
    """Test that read_upload_file_with_limit accepts files at exact size limit."""
    from app.api.v1.openai import read_upload_file_with_limit

    exact_content = b"Z" * 1000  # Exactly at limit
    mock_file = MockUploadFile(exact_content, size=1000)

    result = await read_upload_file_with_limit(mock_file, max_size=1000)
    assert result == exact_content
    assert len(result) == 1000


@pytest.mark.asyncio
async def test_read_upload_file_with_limit_one_byte_over():
    """Test that read_upload_file_with_limit rejects files just one byte over limit."""
    from app.api.v1.openai import read_upload_file_with_limit
    from fastapi import HTTPException

    over_content = b"A" * 1001  # Just one byte over
    mock_file = MockUploadFile(over_content, size=1001)

    with pytest.raises(HTTPException) as exc_info:
        await read_upload_file_with_limit(mock_file, max_size=1000)

    assert exc_info.value.status_code == 413


# ============================================================================
# Audio Transcriptions File Size Limit Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_transcriptions_rejects_large_file():
    """Test that audio transcriptions rejects files exceeding MAX_AUDIO_REQUEST_SIZE."""
    from fastapi import HTTPException

    # Create a mock that raises 413 for large files
    async def mock_read_with_413(*args, **kwargs):
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum: 100 bytes",
        )

    # Create oversized audio content
    large_audio = b"RIFF" + b"\x00" * 500  # Exceeds 100 byte test limit

    with patch(
        "app.api.v1.openai.read_upload_file_with_limit",
        side_effect=mock_read_with_413,
    ):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
            },
            files=[
                ("file", ("test.wav", large_audio, "audio/wav")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    # Should return 413 Payload Too Large
    assert response.status_code == 413
    response_data = response.json()
    assert "detail" in response_data
    assert "File too large" in response_data["detail"]


@pytest.mark.asyncio
async def test_images_edits_rejects_large_image():
    """Test that images edits rejects images exceeding MAX_IMAGE_REQUEST_SIZE."""
    from fastapi import HTTPException

    # Create a mock that raises 413 for large files
    async def mock_read_with_413(*args, **kwargs):
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum: 100 bytes",
        )

    # Create oversized image content
    large_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 500  # Exceeds 100 byte test limit

    with patch(
        "app.api.v1.openai.read_upload_file_with_limit",
        side_effect=mock_read_with_413,
    ):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "Edit this image",
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", large_image, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    # Should return 413 Payload Too Large
    assert response.status_code == 413
    response_data = response.json()
    assert "detail" in response_data
    assert "File too large" in response_data["detail"]


# ============================================================================
# Rerank Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_success(respx_mock):
    """Test successful rerank request."""
    # Test request data
    request_data = {
        "model": "rerank-model",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "London is the capital of the UK.",
        ],
        "top_n": 3,
    }

    # Mock response data
    response_id = "rerank-123"
    response_data = {
        "id": response_id,
        "results": [
            {
                "index": 0,
                "relevance_score": 0.98,
                "document": {"text": "Paris is the capital of France."},
            },
            {
                "index": 2,
                "relevance_score": 0.12,
                "document": {"text": "London is the capital of the UK."},
            },
            {
                "index": 1,
                "relevance_score": 0.05,
                "document": {"text": "Berlin is the capital of Germany."},
            },
        ],
        "model": "rerank-model",
        "usage": {"total_tokens": 50},
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert result["id"] == response_id
        assert len(result["results"]) == 3
        assert result["results"][0]["relevance_score"] == 0.98
        assert result["results"][0]["document"]["text"] == "Paris is the capital of France."

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_with_document_objects(respx_mock):
    """Test rerank request with document objects (containing text field)."""
    request_data = {
        "model": "rerank-model",
        "query": "What is the capital of France?",
        "documents": [
            {"text": "Paris is the capital of France."},
            {"text": "Berlin is the capital of Germany."},
        ],
    }

    response_id = "rerank-456"
    response_data = {
        "id": response_id,
        "results": [
            {
                "index": 0,
                "relevance_score": 0.95,
                "document": {"text": "Paris is the capital of France."},
            },
            {
                "index": 1,
                "relevance_score": 0.10,
                "document": {"text": "Berlin is the capital of Germany."},
            },
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert len(result["results"]) == 2


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_upstream_error(respx_mock):
    """Test rerank request when upstream returns error."""
    request_data = {
        "model": "rerank-model",
        "query": "Test query",
        "documents": ["Doc 1", "Doc 2"],
    }

    error_response = {"error": {"message": "Model not found", "type": "invalid_request_error"}}
    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(404, json=error_response)
    )

    response = client.post(
        "/v1/rerank",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 404
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_with_request_hash(respx_mock):
    """Test rerank request with X-Request-Hash header."""
    request_data = {
        "model": "rerank-model",
        "query": "Test query",
        "documents": ["Doc 1", "Doc 2"],
    }

    expected_hash = "custom-rerank-hash"
    response_id = "rerank-789"
    response_data = {
        "id": response_id,
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_generates_id_if_missing(respx_mock):
    """Test that rerank endpoint generates ID if not in response."""
    request_data = {
        "model": "rerank-model",
        "query": "Test query",
        "documents": ["Doc 1"],
    }

    # Response without ID
    response_data = {
        "results": [{"index": 0, "relevance_score": 0.9}],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("rerank-")
        assert len(result["id"]) == 31  # "rerank-" + 24 hex chars


@pytest.mark.asyncio
async def test_rerank_rejects_oversized_request():
    """Test that rerank endpoint rejects oversized requests."""
    from app.api.v1.openai import read_body_with_limit

    large_query = "x" * 1000
    request_data = {
        "model": "rerank-model",
        "query": large_query,
        "documents": ["Doc 1"],
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 413
    response_data = response.json()
    assert "detail" in response_data
    assert "Request body too large" in response_data["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_without_return_documents(respx_mock):
    """Test rerank request with return_documents=false (no document text in response)."""
    request_data = {
        "model": "rerank-model",
        "query": "Test query",
        "documents": ["Doc 1", "Doc 2"],
        "return_documents": False,
    }

    response_id = "rerank-no-docs"
    response_data = {
        "id": response_id,
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert len(result["results"]) == 2
        # No document field in results
        assert "document" not in result["results"][0]


# ============================================================================
# Score Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_success(respx_mock):
    """Test successful score request."""
    # Test request data
    request_data = {
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "text_1": "What is the capital of France?",
        "text_2": "The capital of France is Paris.",
    }

    # Mock response data
    response_id = "score-123"
    response_data = {
        "id": response_id,
        "score": 0.95,
        "model": "Qwen/Qwen3-Reranker-0.6B",
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/score",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify response content
        result = response.json()
        assert result["id"] == response_id
        assert result["score"] == 0.95
        assert result["model"] == "Qwen/Qwen3-Reranker-0.6B"

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_upstream_error(respx_mock):
    """Test score request when upstream returns error."""
    request_data = {
        "model": "test-model",
        "text_1": "Query text",
        "text_2": "Document text",
    }

    error_response = {"error": {"message": "Model not found", "type": "invalid_request_error"}}
    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(404, json=error_response)
    )

    response = client.post(
        "/v1/score",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 404
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_with_request_hash(respx_mock):
    """Test score request with X-Request-Hash header."""
    request_data = {
        "model": "test-model",
        "text_1": "Query",
        "text_2": "Document",
    }

    expected_hash = "custom-score-hash"
    response_id = "score-789"
    response_data = {
        "id": response_id,
        "score": 0.75,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/score",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_generates_id_if_missing(respx_mock):
    """Test that score endpoint generates ID if not in response."""
    request_data = {
        "model": "test-model",
        "text_1": "Query",
        "text_2": "Document",
    }

    # Response without ID
    response_data = {
        "score": 0.85,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("score-")
        assert len(result["id"]) == 30  # "score-" + 24 hex chars


@pytest.mark.asyncio
async def test_score_rejects_oversized_request():
    """Test that score endpoint rejects oversized requests."""
    from app.api.v1.openai import read_body_with_limit

    large_text = "x" * 1000
    request_data = {
        "model": "test-model",
        "text_1": large_text,
        "text_2": "short",
    }

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        _create_limited_read_body(read_body_with_limit, max_size=100),
    ):
        response = client.post(
            "/v1/score",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 413
    response_data = response.json()
    assert "detail" in response_data
    assert "Request body too large" in response_data["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_with_negative_score(respx_mock):
    """Test score request with negative score value."""
    request_data = {
        "model": "test-model",
        "text_1": "Unrelated query",
        "text_2": "Completely different document",
    }

    response_id = "score-negative"
    response_data = {
        "id": response_id,
        "score": -0.5,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert result["score"] == -0.5