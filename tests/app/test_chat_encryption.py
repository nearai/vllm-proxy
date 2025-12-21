import httpx
import pytest
from fastapi.testclient import TestClient
import json
from hashlib import sha256

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
mock_quote_module = __import__("tests.app.mock_quote", fromlist=[""])
sys.modules["app.quote.quote"] = mock_quote_module

# Replace the mock contexts with real contexts so decryption works
# The app code will use these contexts for decryption
mock_quote_module.ecdsa_context = real_ecdsa_context
mock_quote_module.ed25519_context = real_ed25519_context
mock_quote_module.ecdsa_quote = real_ecdsa_context
mock_quote_module.ed25519_quote = real_ed25519_context

# Now we can safely import app code
from app.main import app
from app.api.v1.openai import VLLM_URL, VLLM_COMPLETIONS_URL

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


# ==================== Chat Completions Endpoint Tests ====================

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
        "/v1/chat/completions",
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
        "/v1/chat/completions",
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
        "/v1/chat/completions",
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
        "/v1/chat/completions",
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
                    assert (
                        len(delta_content) >= 64 or delta_content == ""
                    )  # Empty string is allowed
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
        "/v1/chat/completions",
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
                {
                    "delta": {"content": ""},
                    "index": 0,
                    "finish_reason": None,
                }  # Empty string
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
@pytest.mark.respx
async def test_encrypted_chat_completions_partial_headers(respx_mock):
    """Test that providing only one encryption header results in plain request (encryption disabled)."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
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
    
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Missing X-Signing-Algo - should work as plain request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Pub-Key": "some-key",
        },
    )
    assert response.status_code == 200
    assert route.called
    # Response should be plain text (not encrypted)
    response_json = response.json()
    assert response_json["choices"][0]["message"]["content"] == "Response"

    # Missing X-Signing-Pub-Key - should work as plain request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
        },
    )
    assert response.status_code == 200
    # Response should be plain text (not encrypted)
    response_json = response.json()
    assert response_json["choices"][0]["message"]["content"] == "Response"


@pytest.mark.asyncio
async def test_encrypted_chat_completions_invalid_algo():
    """Test encrypted chat completions with invalid signing algorithm."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response = client.post(
        "/v1/chat/completions",
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
    """Test that plain text content (not encrypted) is passed through when encryption headers are provided."""
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
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Signing-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )
    
    # Should succeed - plain text is treated as plain text (not decrypted, but response will be encrypted)
    assert response.status_code == 200
    assert route.called
    # Response should be encrypted since encryption headers were provided
    response_json = response.json()
    response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(response_content, str)
    assert len(response_content) >= 64  # Should be encrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_plain_request_no_encryption(respx_mock):
    """Test that plain requests without encryption headers work correctly."""
    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, plain text"}],
        "stream": False,
    }
    
    # Mock response
    chat_id = "chatcmpl-plain-no-enc-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Plain response"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    
    # Setup RESPX mock
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )
    
    # Request without encryption headers
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )
    
    # Should succeed with plain text (no encryption)
    assert response.status_code == 200
    assert route.called
    response_json = response.json()
    # Response should be plain text (not encrypted)
    assert response_json["choices"][0]["message"]["content"] == "Plain response"


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
        "/v1/chat/completions",
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
        "/v1/chat/completions",
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
        "/v1/chat/completions",
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


@pytest.mark.asyncio
@pytest.mark.respx
async def test_signature_encrypted_non_streaming_ecdsa(respx_mock):
    """Test signature endpoint after encrypted non-streaming chat completion with ECDSA."""
    # Encrypt the request content
    plain_content = "Hello, how are you?"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    # Mock non-streaming response data
    chat_id = "chatcmpl-signature-ecdsa-123"
    plain_response_content = "I'm doing well, thank you!"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": plain_response_content,
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

    # Make encrypted chat completion request
    response = client.post(
        "/v1/chat/completions",
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
    assert response_json["id"] == chat_id

    # Verify content is encrypted in response
    message = response_json["choices"][0]["message"]
    assert isinstance(message["content"], str)
    assert len(message["content"]) >= 64  # Should be encrypted
    assert message["content"] != plain_response_content

    # Calculate expected hashes (Option A: hash encrypted request/response bodies)
    # Request hash should be of original encrypted request body (what client sends)
    # Note: TestClient sends JSON in compact format (no spaces), so we use separators
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    # Response hash should be of encrypted response (what client receives)
    # Manually encrypt the response content to match what the server sends
    encrypted_response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": message["content"],  # Use the encrypted content from actual response
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    # Note: json.dumps() with separators matches the server's compact format
    encrypted_response_body = json.dumps(encrypted_response_data, separators=(",", ":")).encode("utf-8")
    expected_response_hash = sha256(encrypted_response_body).hexdigest()

    # Fetch signature
    signature_response = client.get(
        f"/v1/signature/{chat_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify signature response
    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert signature_data["signing_algo"] == ECDSA
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
    assert signature_data["signature"].startswith("0x")
    assert len(signature_data["signature"]) > 0
    assert signature_data["signing_address"] is not None


@pytest.mark.asyncio
@pytest.mark.respx
async def test_signature_encrypted_non_streaming_ed25519(respx_mock):
    """Test signature endpoint after encrypted non-streaming chat completion with Ed25519."""
    # Encrypt the request content
    plain_content = "What is the weather?"
    encrypted_content = encrypt_content(plain_content, ED25519)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    # Mock non-streaming response data
    chat_id = "chatcmpl-signature-ed25519-123"
    plain_response_content = "I don't have access to weather data."
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": plain_response_content,
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

    # Make encrypted chat completion request
    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Signing-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )

    # Verify response
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["id"] == chat_id

    # Get the encrypted content from the actual response
    encrypted_response_content = response_json["choices"][0]["message"]["content"]

    # Calculate expected hashes (Option A: hash encrypted request/response bodies)
    # Request hash should be of original encrypted request body (what client sends)
    # Note: TestClient sends JSON in compact format (no spaces), so we use separators
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    # Response hash should be of encrypted response (what client receives)
    encrypted_response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": encrypted_response_content,  # Use encrypted content from actual response
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }
    # Note: json.dumps() with separators matches the server's compact format
    encrypted_response_body = json.dumps(encrypted_response_data, separators=(",", ":")).encode("utf-8")
    expected_response_hash = sha256(encrypted_response_body).hexdigest()

    # Fetch signature with explicit Ed25519 algorithm
    signature_response = client.get(
        f"/v1/signature/{chat_id}?signing_algo={ED25519}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify signature response
    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert signature_data["signing_algo"] == ED25519
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
    # Ed25519 signatures don't have "0x" prefix (unlike ECDSA)
    assert len(signature_data["signature"]) > 0
    # Verify it's a valid hex string (Ed25519 signatures are 128 hex characters = 64 bytes)
    assert all(c in "0123456789abcdef" for c in signature_data["signature"].lower())
    assert len(signature_data["signature"]) == 128  # Ed25519 signature is 64 bytes = 128 hex chars
    assert signature_data["signing_address"] is not None


@pytest.mark.asyncio
@pytest.mark.respx
async def test_signature_encrypted_streaming_ecdsa(respx_mock):
    """Test signature endpoint after encrypted streaming chat completion with ECDSA."""
    # Encrypt the request content
    plain_content = "Tell me a story"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }

    # Mock streaming response chunks
    chat_id = "chatcmpl-signature-stream-ecdsa-123"
    chunks = [
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Once"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": " upon"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": " a time"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"finish_reason": "stop"}]},
    ]

    # Setup RESPX mock for streaming
    # Need to manually add [DONE] marker
    async def stream_generator():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=stream_generator(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Make encrypted streaming chat completion request
    response = client.post(
        "/v1/chat/completions",
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

    # Calculate expected hashes (Option A: hash encrypted request/response bodies)
    # Request hash should be of original encrypted request body (what client sends)
    # Note: TestClient sends JSON in compact format (no spaces), so we use separators
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    # Response hash should be of encrypted chunks (what client receives)
    expected_response_hash = sha256(response.content).hexdigest()

    # Fetch signature
    signature_response = client.get(
        f"/v1/signature/{chat_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify signature response
    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert signature_data["signing_algo"] == ECDSA
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
    assert signature_data["signature"].startswith("0x")
    assert len(signature_data["signature"]) > 0
    assert signature_data["signing_address"] is not None


@pytest.mark.asyncio
@pytest.mark.respx
async def test_signature_encrypted_streaming_ed25519(respx_mock):
    """Test signature endpoint after encrypted streaming chat completion with Ed25519."""
    # Encrypt the request content
    plain_content = "Count to 3"
    encrypted_content = encrypt_content(plain_content, ED25519)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }

    # Mock streaming response chunks
    chat_id = "chatcmpl-signature-stream-ed25519-123"
    chunks = [
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": "1"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": ", "}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": "2"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": ", "}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": "3"}}]},
        {"id": chat_id, "object": "chat.completion.chunk", "choices": [{"finish_reason": "stop"}]},
    ]

    # Setup RESPX mock for streaming
    async def stream_generator():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    
    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=stream_generator(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Make encrypted streaming chat completion request
    response = client.post(
        "/v1/chat/completions",
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

    # Calculate expected hashes (Option A: hash encrypted request/response bodies)
    # Request hash should be of original encrypted request body (what client sends)
    # Note: TestClient sends JSON in compact format (no spaces), so we use separators
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    # Response hash should be of encrypted chunks (what client receives)
    expected_response_hash = sha256(response.content).hexdigest()

    # Fetch signature with explicit Ed25519 algorithm
    signature_response = client.get(
        f"/v1/signature/{chat_id}?signing_algo={ED25519}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify signature response
    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert signature_data["signing_algo"] == ED25519
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
    # Ed25519 signatures don't have "0x" prefix (unlike ECDSA)
    assert len(signature_data["signature"]) > 0
    # Verify it's a valid hex string (Ed25519 signatures are 128 hex characters = 64 bytes)
    assert all(c in "0123456789abcdef" for c in signature_data["signature"].lower())
    assert len(signature_data["signature"]) == 128  # Ed25519 signature is 64 bytes = 128 hex chars
    assert signature_data["signing_address"] is not None


# ==================== Completions Endpoint Tests ====================

@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_completions_non_streaming_ecdsa(respx_mock):
    """Test encrypted completions with ECDSA, non-streaming."""
    # Encrypt the request prompt
    plain_prompt = "Hello, how are you?"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    request_data = {
        "model": "test-model",
        "prompt": encrypted_prompt,
        "stream": False,
    }

    # Mock non-streaming response data
    completion_id = "cmpl-completion-ecdsa-123"
    plain_response_text = "I'm doing well, thank you!"
    response_data = {
        "id": completion_id,
        "object": "text_completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "text": plain_response_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/completions",
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

    # Verify text is encrypted in response
    choice = response_json["choices"][0]
    assert isinstance(choice["text"], str)
    assert len(choice["text"]) >= 64  # Should be encrypted
    assert choice["text"] != plain_response_text
    assert all(c in "0123456789abcdefABCDEF" for c in choice["text"])

    # Verify prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["prompt"] == plain_prompt  # Should be decrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_completions_non_streaming_ed25519(respx_mock):
    """Test encrypted completions with Ed25519, non-streaming."""
    # Encrypt the request prompt
    plain_prompt = "What is the weather?"
    encrypted_prompt = encrypt_content(plain_prompt, ED25519)

    request_data = {
        "model": "test-model",
        "prompt": encrypted_prompt,
        "stream": False,
    }

    # Mock non-streaming response data
    completion_id = "cmpl-completion-ed25519-123"
    plain_response_text = "I don't have access to weather data."
    response_data = {
        "id": completion_id,
        "object": "text_completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "text": plain_response_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/completions",
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

    # Verify text is encrypted in response
    choice = response_json["choices"][0]
    assert isinstance(choice["text"], str)
    assert len(choice["text"]) >= 64  # Should be encrypted
    assert choice["text"] != plain_response_text

    # Verify prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["prompt"] == plain_prompt  # Should be decrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_completions_streaming_ecdsa(respx_mock):
    """Test encrypted completions with ECDSA, streaming."""
    # Encrypt the request prompt
    plain_prompt = "Tell me a story"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    request_data = {
        "model": "test-model",
        "prompt": encrypted_prompt,
        "stream": True,
    }

    # Mock streaming response chunks
    completion_id = "cmpl-completion-stream-ecdsa-123"
    chunks = [
        {"id": completion_id, "object": "text_completion", "choices": [{"text": "Once", "index": 0}]},
        {"id": completion_id, "object": "text_completion", "choices": [{"text": " upon", "index": 0}]},
        {"id": completion_id, "object": "text_completion", "choices": [{"text": " a time", "index": 0}]},
        {"id": completion_id, "object": "text_completion", "choices": [{"finish_reason": "stop", "index": 0}]},
    ]

    # Setup RESPX mock for streaming
    async def stream_generator():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    
    route = respx_mock.post(VLLM_COMPLETIONS_URL).mock(
        return_value=httpx.Response(
            200,
            stream=stream_generator(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    # Make encrypted streaming completions request
    response = client.post(
        "/v1/completions",
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
    content = response.content.decode()
    chunks_received = []
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunks_received.append(json.loads(data))

    # Verify chunks are encrypted
    assert len(chunks_received) > 0
    for chunk in chunks_received:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "text" in choice and choice["text"]:
                # Text should be encrypted (hex string)
                assert isinstance(choice["text"], str)
                assert len(choice["text"]) >= 64  # Should be encrypted
                assert all(c in "0123456789abcdefABCDEF" for c in choice["text"])

    # Verify prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["prompt"] == plain_prompt  # Should be decrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_completions_plain_request_no_encryption(respx_mock):
    """Test that plain completions requests work without encryption headers."""
    request_data = {
        "model": "test-model",
        "prompt": "Hello, world!",
        "stream": False,
    }

    # Mock non-streaming response data
    completion_id = "cmpl-plain-123"
    response_data = {
        "id": completion_id,
        "object": "text_completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "text": "Response text",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request without encryption headers
    response = client.post(
        "/v1/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify text is NOT encrypted (plain text)
    choice = response_json["choices"][0]
    assert choice["text"] == "Response text"  # Should be plain text
