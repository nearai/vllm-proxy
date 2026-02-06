from unittest.mock import patch
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
from app.api.v1.openai import VLLM_BASE_URL

client = TestClient(app)


# ============================================================================
# Non-Streaming JSON Response Tests (with signature)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_json_response_with_id(respx_mock):
    """Test passthrough returns JSON response and caches signature when response has id."""
    response_data = {
        "id": "test-123",
        "result": "some data",
    }

    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/some/new/endpoint").mock(
        return_value=httpx.Response(
            200,
            json=response_data,
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/some/new/endpoint",
            json={"query": "test"},
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert result["id"] == "test-123"
        assert result["result"] == "some data"

        # Verify signature was cached
        mock_cache.set_chat.assert_called_once()
        call_args = mock_cache.set_chat.call_args
        assert call_args[0][0] == "test-123"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_json_response_without_id(respx_mock):
    """Test passthrough generates and injects id when response lacks one."""
    response_data = {
        "result": "some data",
    }

    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/custom/endpoint").mock(
        return_value=httpx.Response(
            200,
            json=response_data,
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/custom/endpoint",
            json={"query": "test"},
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        # Verify ID was generated with correct prefix
        assert result["id"].startswith("passthrough-")
        assert len(result["id"]) == 36  # "passthrough-" + 24 hex chars

        # Verify signature was cached
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_json_with_request_hash(respx_mock):
    """Test passthrough uses X-Request-Hash header when provided."""
    response_data = {"id": "test-hash", "result": "ok"}
    expected_hash = "custom-passthrough-hash"

    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/test/endpoint").mock(
        return_value=httpx.Response(
            200,
            json=response_data,
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/test/endpoint",
            json={"query": "test"},
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Request-Hash": expected_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_any_call(
            f"Passthrough: Using client-provided request hash: {expected_hash}"
        )

        mock_cache.set_chat.assert_called_once()


# ============================================================================
# Streaming SSE Response Tests (with signature)
# ============================================================================


async def yield_sse_response(data_list):
    for data in data_list:
        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_streaming_with_id(respx_mock):
    """Test streaming passthrough extracts id from first chunk and caches signature."""
    chat_id = "stream-123"
    responses = [
        {"id": chat_id, "choices": [{"delta": {"content": "Hello"}}]},
        {"id": chat_id, "choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/streaming/endpoint").mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/streaming/endpoint",
            json={"stream": True},
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        # Verify signature was cached with extracted id
        mock_cache.set_chat.assert_called_once()
        call_args = mock_cache.set_chat.call_args
        assert call_args[0][0] == chat_id


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_streaming_without_id(respx_mock):
    """Test streaming passthrough generates id and appends final event when no id in chunks."""
    responses = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/streaming/no-id").mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(responses),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/v1/streaming/no-id",
            json={"stream": True},
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        # Verify signature was cached with generated id
        mock_cache.set_chat.assert_called_once()
        call_args = mock_cache.set_chat.call_args
        assert call_args[0][0].startswith("passthrough-")

        # Verify final SSE event was appended with generated id
        content = response.content.decode()
        lines = [l for l in content.split("\n") if l.startswith("data: ")]
        last_data = json.loads(lines[-1].replace("data: ", ""))
        assert last_data["id"].startswith("passthrough-")


# ============================================================================
# Non-JSON Response Tests (no signature)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_plain_text_response(respx_mock):
    """Test passthrough returns plain text response without signature."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/v1/health").mock(
        return_value=httpx.Response(
            200,
            text="OK",
            headers={"Content-Type": "text/plain"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.get(
            "/v1/health",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert response.text == "OK"
        assert route.called

        # No signature for non-JSON responses
        mock_cache.set_chat.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_invalid_json_response(respx_mock):
    """Test passthrough handles application/json with invalid JSON body."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/v1/broken").mock(
        return_value=httpx.Response(
            200,
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.get(
            "/v1/broken",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert response.content == b"not valid json"
        assert route.called

        # No signature when JSON parsing fails
        mock_cache.set_chat.assert_not_called()


# ============================================================================
# Error Response Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_upstream_error(respx_mock):
    """Test passthrough returns generic error for upstream failures."""
    route = respx_mock.post(f"{VLLM_BASE_URL}/v1/failing/endpoint").mock(
        return_value=httpx.Response(
            500,
            json={"error": "internal details that should not leak"},
        )
    )

    response = client.post(
        "/v1/failing/endpoint",
        json={"query": "test"},
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 500
    assert route.called

    # Verify generic error message (not upstream details)
    response_data = response.json()
    assert "internal details" not in json.dumps(response_data)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_upstream_404(respx_mock):
    """Test passthrough forwards 404 status from upstream."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/v1/nonexistent").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )

    response = client.get(
        "/v1/nonexistent",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 404
    assert route.called


@pytest.mark.asyncio
async def test_passthrough_requires_auth():
    """Test passthrough requires authorization header."""
    response = client.get("/v1/any/path")

    assert response.status_code == 401


# ============================================================================
# Path Traversal Protection Tests
# ============================================================================

# Note: Starlette normalizes ".." segments in URLs before they reach handlers,
# so we test handle_passthrough directly for path traversal defense-in-depth.


@pytest.mark.asyncio
async def test_passthrough_blocks_path_traversal_direct():
    """Test handle_passthrough blocks .. in path segments (defense-in-depth)."""
    from app.api.v1.openai import handle_passthrough
    from fastapi import HTTPException
    from unittest.mock import MagicMock

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.path = "/v1/../../secret"
    mock_request.url.query = ""

    with pytest.raises(HTTPException) as exc_info:
        await handle_passthrough(
            mock_request, "http://backend/v1/../../secret", "../../secret"
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_passthrough_blocks_encoded_traversal_direct():
    """Test handle_passthrough blocks URL-encoded path traversal (defense-in-depth)."""
    from app.api.v1.openai import handle_passthrough
    from fastapi import HTTPException
    from unittest.mock import MagicMock

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.path = "/v1/%2e%2e/secret"
    mock_request.url.query = ""

    # Path would be URL-decoded by Starlette to "../secret", but if something
    # passes "%2e%2e" as the path param, unquote catches it
    with pytest.raises(HTTPException) as exc_info:
        await handle_passthrough(
            mock_request, "http://backend/v1/%2e%2e/secret", "%2e%2e/secret"
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_allows_valid_dotted_paths(respx_mock):
    """Test passthrough allows paths with dots that aren't traversal (e.g. v2.0)."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/v1/api/v2.0/status").mock(
        return_value=httpx.Response(
            200,
            json={"status": "ok", "id": "dot-test"},
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache"):
        response = client.get(
            "/v1/api/v2.0/status",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


# ============================================================================
# Query String Forwarding Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_forwards_query_params(respx_mock):
    """Test passthrough forwards query parameters to backend."""
    route = respx_mock.get(
        f"{VLLM_BASE_URL}/v1/search?q=test&limit=10"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"results": [], "id": "query-test"},
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache"):
        response = client.get(
            "/v1/search?q=test&limit=10",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


# ============================================================================
# Root-Level Passthrough Tests (non-/v1/ paths)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_root_passthrough_json_response(respx_mock):
    """Test root-level passthrough for non-/v1/ endpoints."""
    response_data = {"id": "root-123", "tokens": [1, 2, 3]}

    route = respx_mock.post(f"{VLLM_BASE_URL}/tokenize").mock(
        return_value=httpx.Response(
            200,
            json=response_data,
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.post(
            "/tokenize",
            json={"text": "hello"},
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called

        result = response.json()
        assert result["id"] == "root-123"

        # Verify signature was cached
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_root_passthrough_plain_text(respx_mock):
    """Test root-level passthrough for plain text responses (e.g. metrics)."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/some/metrics").mock(
        return_value=httpx.Response(
            200,
            text="# HELP counter\ncounter 42",
            headers={"Content-Type": "text/plain"},
        )
    )

    with patch("app.api.v1.openai.cache") as mock_cache:
        response = client.get(
            "/some/metrics",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert "counter 42" in response.text
        assert route.called

        mock_cache.set_chat.assert_not_called()


@pytest.mark.asyncio
async def test_root_passthrough_blocks_path_traversal_direct():
    """Test handle_passthrough blocks path traversal for root-level routes (defense-in-depth)."""
    from app.api.v1.openai import handle_passthrough
    from fastapi import HTTPException
    from unittest.mock import MagicMock

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.path = "/../../etc/passwd"
    mock_request.url.query = ""

    with pytest.raises(HTTPException) as exc_info:
        await handle_passthrough(
            mock_request, "http://backend/../../etc/passwd", "../../etc/passwd"
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_root_passthrough_requires_auth():
    """Test root-level passthrough requires authorization."""
    response = client.post("/some/endpoint", json={"test": True})

    assert response.status_code == 401


# ============================================================================
# HTTP Method Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_get_request(respx_mock):
    """Test passthrough handles GET requests."""
    route = respx_mock.get(f"{VLLM_BASE_URL}/v1/status").mock(
        return_value=httpx.Response(
            200,
            json={"status": "ok", "id": "get-test"},
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache"):
        response = client.get(
            "/v1/status",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_passthrough_delete_request(respx_mock):
    """Test passthrough handles DELETE requests."""
    route = respx_mock.delete(f"{VLLM_BASE_URL}/v1/resource/123").mock(
        return_value=httpx.Response(
            200,
            json={"deleted": True, "id": "del-test"},
            headers={"Content-Type": "application/json"},
        )
    )

    with patch("app.api.v1.openai.cache"):
        response = client.delete(
            "/v1/resource/123",
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200
        assert route.called


# ============================================================================
# Request Size Limit Tests
# ============================================================================


@pytest.mark.asyncio
async def test_passthrough_rejects_oversized_request():
    """Test passthrough enforces request size limit."""
    from app.api.v1.openai import read_body_with_limit
    from fastapi import HTTPException

    # Create oversized payload
    large_data = {"data": "x" * 1000}

    async def limited_read_body(request, max_size=None):
        return await read_body_with_limit(request, max_size=100)

    with patch(
        "app.api.v1.openai.read_body_with_limit",
        limited_read_body,
    ):
        response = client.post(
            "/v1/some/endpoint",
            json=large_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 413


# ============================================================================
# Existing Route Priority Tests
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_defined_routes_not_caught_by_passthrough(respx_mock):
    """Test that explicitly defined routes take priority over passthrough."""
    from app.api.v1.openai import VLLM_MODELS_URL

    # Mock the models endpoint (explicitly defined)
    route = respx_mock.get(VLLM_MODELS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "test-model"}]},
        )
    )

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert route.called
    result = response.json()
    assert "data" in result


@pytest.mark.asyncio
async def test_root_route_not_caught_by_passthrough():
    """Test that / root route is not caught by root-level passthrough."""
    response = client.get("/")

    assert response.status_code == 200
    # Root returns ok() response, not a passthrough


@pytest.mark.asyncio
async def test_version_route_not_caught_by_passthrough():
    """Test that /version route is not caught by root-level passthrough."""
    response = client.get("/version")

    assert response.status_code == 200
    result = response.json()
    assert "version" in result
    assert "type" in result
