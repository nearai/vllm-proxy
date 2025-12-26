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
from app.api.v1.openai import VLLM_URL, VLLM_BASE_URL
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
    """Create a wrapper that calls read_body_with_limit with a custom max_size."""

    async def limited_read_body(request):
        return await original_func(request, max_size=max_size)

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
