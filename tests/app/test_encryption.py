from unittest.mock import patch, AsyncMock
import httpx
import pytest
from fastapi.testclient import TestClient
import json

# Import and setup test environment before importing app
from tests.app.test_helpers import setup_test_environment, TEST_AUTH_HEADER

# Setup all mocks before importing app
setup_test_environment()

# Import real quote contexts BEFORE replacing with mock (needed for encryption)
import sys
import app.quote.quote as real_quote
from app.encryption.encryption import encrypt_data

# Store real contexts for encryption tests
real_ecdsa_context = real_quote.ecdsa_context
real_ed25519_context = real_quote.ed25519_context
ECDSA = real_quote.ECDSA
ED25519 = real_quote.ED25519

# Now replace the quote module with our mock for the rest of the app
sys.modules["app.quote.quote"] = __import__("tests.app.mock_quote", fromlist=[""])

# Now we can safely import app code
from app.main import app
from app.api.v1.openai import VLLM_URL

client = TestClient(app)


async def yield_sse_response(data_list):
    for data in data_list:
        yield f"data: {json.dumps(data)}\n\n".encode("utf-8")


def encrypt_content(content: str, signing_algo: str) -> str:
    """Helper to encrypt content using the server's public key."""
    if signing_algo == ECDSA:
        public_key = real_ecdsa_context.signing_public_key
    else:
        public_key = real_ed25519_context.signing_public_key
    
    encrypted_data = encrypt_data(content.encode("utf-8"), public_key, signing_algo)
    return encrypted_data.hex()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_non_streaming_ecdsa(respx_mock):
    """Test encrypted chat completions with ECDSA, non-streaming."""
    # Encrypt the request content
    plain_content = "Hello, how are you?"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }
    
    # Mock non-streaming response data
    chat_id = "chatcmpl-encrypted-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you!",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request with encryption headers
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()
    
    # Verify response structure
    assert response_json["id"] == chat_id
    assert "choices" in response_json
    assert len(response_json["choices"]) > 0
    
    # Verify response content is encrypted (hex string)
    response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(response_content, str)
    assert len(response_content) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in response_content)
    
    # Verify the request was decrypted before sending to vLLM
    # Check that vLLM received plain text content
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"] == plain_content


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_non_streaming_ed25519(respx_mock):
    """Test encrypted chat completions with Ed25519, non-streaming."""
    # Encrypt the request content
    plain_content = "What is the weather today?"
    encrypted_content = encrypt_content(plain_content, ED25519)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }
    
    # Mock non-streaming response data
    chat_id = "chatcmpl-encrypted-456"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I don't have access to real-time weather data.",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request with encryption headers
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Signing-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()
    
    # Verify response content is encrypted
    response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(response_content, str)
    assert len(response_content) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in response_content)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_with_reasoning_content(respx_mock):
    """Test encrypted chat completions with reasoning_content field."""
    # Encrypt the request content
    plain_content = "Solve this math problem: 2+2"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }
    
    # Mock response with reasoning_content
    chat_id = "chatcmpl-reasoning-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "I need to add 2 and 2 together. 2 + 2 = 4.",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    response_json = response.json()
    
    # Verify both content and reasoning_content are encrypted
    message = response_json["choices"][0]["message"]
    assert isinstance(message["content"], str)
    assert len(message["content"]) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in message["content"])
    
    assert isinstance(message["reasoning_content"], str)
    assert len(message["reasoning_content"]) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in message["reasoning_content"])


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_streaming(respx_mock):
    """Test encrypted chat completions with streaming response."""
    # Encrypt the request content
    plain_content = "Tell me a story"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }
    
    # Mock streaming response data
    chat_id = "chatcmpl-stream-123"
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
                {"delta": {"content": "Once"}, "index": 0, "finish_reason": None}
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"content": " upon"}, "index": 0, "finish_reason": None}
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
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    assert route.called
    
    # Collect streaming chunks
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunk = json.loads(data)
                chunks.append(chunk)
    
    # Verify chunks are encrypted
    assert len(chunks) > 0
    for chunk in chunks:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                delta_content = choice["delta"]["content"]
                if delta_content:
                    # Content should be encrypted (hex string)
                    assert isinstance(delta_content, str)
                    assert len(delta_content) >= 64 or delta_content == ""  # Empty string is allowed
                    if delta_content:
                        assert all(c in "0123456789abcdefABCDEF" for c in delta_content)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_streaming_with_reasoning(respx_mock):
    """Test encrypted streaming with reasoning_content."""
    # Encrypt the request content
    plain_content = "Think step by step: what is 5*5?"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }
    
    # Mock streaming response with reasoning_content
    chat_id = "chatcmpl-reasoning-stream-123"
    responses = [
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "I need to multiply 5 by 5.",
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": 1677825464,
            "model": "test-model",
            "choices": [
                {"delta": {"content": "25"}, "index": 0, "finish_reason": None}
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
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    
    # Collect streaming chunks
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunk = json.loads(data)
                chunks.append(chunk)
    
    # Verify reasoning_content is encrypted in chunks
    found_reasoning = False
    for chunk in chunks:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "delta" in choice and "reasoning_content" in choice["delta"]:
                reasoning = choice["delta"]["reasoning_content"]
                if reasoning:
                    found_reasoning = True
                    assert isinstance(reasoning, str)
                    assert len(reasoning) >= 64
                    assert all(c in "0123456789abcdefABCDEF" for c in reasoning)
    
    assert found_reasoning, "Should have found encrypted reasoning_content"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_streaming_empty_content(respx_mock):
    """Test that empty string content in streaming responses is not encrypted."""
    # Encrypt the request content
    plain_content = "Test"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }
    
    # Mock streaming response with empty content in one chunk
    chat_id = "chatcmpl-empty-stream-123"
    responses = [
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
            "choices": [
                {"delta": {"content": ""}, "index": 0, "finish_reason": None}  # Empty string
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
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    
    # Collect streaming chunks
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunk = json.loads(data)
                chunks.append(chunk)
    
    # Verify chunks
    assert len(chunks) > 0
    found_empty = False
    found_encrypted = False
    for chunk in chunks:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                delta_content = choice["delta"]["content"]
                if delta_content == "":
                    found_empty = True
                    # Empty string should remain empty, not encrypted
                    assert delta_content == ""
                elif delta_content:
                    found_encrypted = True
                    # Non-empty content should be encrypted
                    assert isinstance(delta_content, str)
                    assert len(delta_content) >= 64
                    assert all(c in "0123456789abcdefABCDEF" for c in delta_content)
    
    assert found_empty, "Should have found empty string content"
    assert found_encrypted, "Should have found encrypted non-empty content"


@pytest.mark.asyncio
async def test_encrypted_chat_completions_missing_headers():
    """Test encrypted chat completions with missing required headers."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    
    # Missing X-Signing-Algo
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )
    assert response.status_code == 422  # Validation error
    
    # Missing X-Signing-Pub-Key
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
        },
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_encrypted_chat_completions_invalid_algo():
    """Test encrypted chat completions with invalid signing algorithm."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "invalid-algo",
            "X-Signing-Pub-Key": "some-key",
        },
    )
    
    assert response.status_code == 400
    response_json = response.json()
    assert "Invalid X-Signing-Algo" in response_json["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_plain_text_content(respx_mock):
    """Test that plain text content (not encrypted) is passed through."""
    # Use plain text content (not encrypted hex string)
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "This is plain text, not encrypted"}],
        "stream": False,
    }
    
    # Mock response
    chat_id = "chatcmpl-plain-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Response"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Should succeed - plain text is treated as plain text
    assert response.status_code == 200
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_multiple_messages(respx_mock):
    """Test encrypted chat completions with multiple messages."""
    # Encrypt multiple messages
    encrypted_content1 = encrypt_content("Hello", ECDSA)
    encrypted_content2 = encrypt_content("How are you?", ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": encrypted_content1},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": encrypted_content2},
        ],
        "stream": False,
    }
    
    # Mock response
    chat_id = "chatcmpl-multi-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "I'm doing well!"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    assert route.called
    
    # Verify request was decrypted
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"] == "Hello"
    assert sent_data["messages"][1]["content"] == "Hi there!"  # Plain text unchanged
    assert sent_data["messages"][2]["content"] == "How are you?"
    
    # Verify response is encrypted
    response_json = response.json()
    response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(response_content, str)
    assert len(response_content) >= 64


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_empty_string_content(respx_mock):
    """Test that empty string content is not encrypted/decrypted."""
    # Test with empty string in request
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": ""}],  # Empty string
        "stream": False,
    }
    
    # Mock response with empty string content
    chat_id = "chatcmpl-empty-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",  # Empty string should not be encrypted
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    assert route.called
    
    # Verify empty string in request was not decrypted (remains empty)
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"] == ""  # Should remain empty
    
    # Verify empty string in response was not encrypted (remains empty)
    response_json = response.json()
    response_content = response_json["choices"][0]["message"]["content"]
    assert response_content == ""  # Should remain empty, not encrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_chat_completions_empty_reasoning_content(respx_mock):
    """Test that empty string reasoning_content is not encrypted."""
    # Encrypt the request content
    plain_content = "Test question"
    encrypted_content = encrypt_content(plain_content, ECDSA)
    
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }
    
    # Mock response with empty reasoning_content
    chat_id = "chatcmpl-empty-reasoning-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                    "reasoning_content": "",  # Empty string should not be encrypted
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Make request
    response = client.post(
        "/v1/encrypted/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Verify response
    assert response.status_code == 200
    response_json = response.json()
    
    # Verify content is encrypted (non-empty)
    message = response_json["choices"][0]["message"]
    assert isinstance(message["content"], str)
    assert len(message["content"]) >= 64  # Should be encrypted
    
    # Verify empty reasoning_content is NOT encrypted (remains empty)
    assert "reasoning_content" in message
    assert message["reasoning_content"] == ""  # Should remain empty, not encrypted

