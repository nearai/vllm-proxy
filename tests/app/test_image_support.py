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
mock_quote_module.ecdsa_context = real_ecdsa_context
mock_quote_module.ed25519_context = real_ed25519_context
mock_quote_module.ecdsa_quote = real_ecdsa_context
mock_quote_module.ed25519_quote = real_ed25519_context

# Now we can safely import app code
from app.main import app
from app.api.v1.openai import VLLM_URL, VLLM_IMAGES_URL

client = TestClient(app)


def encrypt_content(content: str, signing_algo: str) -> str:
    """Helper to encrypt content using the server's public key."""
    if signing_algo == ECDSA:
        public_key = real_ecdsa_context.signing_public_key
    else:
        public_key = real_ed25519_context.signing_public_key

    encrypted_data = encrypt_data(content.encode("utf-8"), public_key, signing_algo)
    return encrypted_data.hex()


# ==================== Image Input (Vision) Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_image_url_no_encryption(respx_mock):
    """Test chat completions with image URL input without encryption."""
    request_data = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ],
        "stream": False,
    }

    # Mock response
    chat_id = "chatcmpl-vision-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I can see a beautiful landscape.",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()
    assert response_json["id"] == chat_id
    # Content should be plain text (not encrypted)
    assert response_json["choices"][0]["message"]["content"] == "I can see a beautiful landscape."


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_base64_image_no_encryption(respx_mock):
    """Test chat completions with base64 image input without encryption."""
    # Simulated base64 image data (just a short string for testing)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    request_data = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "stream": False,
    }

    chat_id = "chatcmpl-base64-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a tiny image."},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert route.called

    # Verify the request was forwarded with the image content intact
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"][0]["type"] == "text"
    assert sent_data["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_multimodal_encrypted(respx_mock):
    """Test chat completions with encrypted multimodal content (text + image)."""
    # Build multimodal content
    multimodal_content = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]

    # Serialize to JSON and encrypt
    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-multimodal-enc-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "I see a cat!"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert route.called

    # Verify the request was decrypted and parsed correctly
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)

    # Content should be decrypted to the original array
    assert isinstance(sent_data["messages"][0]["content"], list)
    assert len(sent_data["messages"][0]["content"]) == 2
    assert sent_data["messages"][0]["content"][0]["type"] == "text"
    assert sent_data["messages"][0]["content"][0]["text"] == "What's in this image?"
    assert sent_data["messages"][0]["content"][1]["type"] == "image_url"

    # Response content should be encrypted
    response_json = response.json()
    encrypted_response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(encrypted_response_content, str)
    assert len(encrypted_response_content) >= 64  # Should be encrypted hex
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_response_content)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_multimodal_encrypted_ed25519(respx_mock):
    """Test chat completions with encrypted multimodal content using Ed25519."""
    multimodal_content = [
        {"type": "text", "text": "Analyze this image"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==", "detail": "high"},
        },
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ED25519)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-multimodal-ed25519-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Analysis complete."},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert route.called

    # Verify decryption worked
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert isinstance(sent_data["messages"][0]["content"], list)
    assert sent_data["messages"][0]["content"][0]["text"] == "Analyze this image"


# ==================== Image Generation Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_generations_no_encryption(respx_mock):
    """Test image generation endpoint without encryption."""
    request_data = {
        "model": "dall-e-3",
        "prompt": "A beautiful sunset over the ocean",
        "n": 1,
        "size": "1024x1024",
    }

    # Mock response with base64 image
    response_data = {
        "created": 1713833628,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUg==",
                "revised_prompt": "A stunning sunset over a calm ocean with vibrant colors",
            }
        ],
    }

    route = respx_mock.post(VLLM_IMAGES_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response structure
    assert "data" in response_json
    assert len(response_json["data"]) == 1
    assert response_json["data"][0]["b64_json"] == "iVBORw0KGgoAAAANSUhEUg=="
    assert "revised_prompt" in response_json["data"][0]

    # Verify an ID was generated
    assert "id" in response_json
    assert response_json["id"].startswith("img-")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_generations_with_encryption(respx_mock):
    """Test image generation endpoint with encryption."""
    plain_prompt = "A cat sitting on a rainbow"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    request_data = {
        "model": "dall-e-3",
        "prompt": encrypted_prompt,
        "n": 1,
        "size": "1024x1024",
    }

    response_data = {
        "created": 1713833628,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
                "revised_prompt": "A colorful cat sitting on a rainbow",
            }
        ],
    }

    route = respx_mock.post(VLLM_IMAGES_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert route.called

    # Verify prompt was decrypted before forwarding
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["prompt"] == plain_prompt

    # Verify response fields are encrypted
    response_json = response.json()
    assert "data" in response_json
    assert len(response_json["data"]) == 1

    # b64_json should be encrypted (hex string)
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert isinstance(encrypted_b64, str)
    assert len(encrypted_b64) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # revised_prompt should also be encrypted
    encrypted_revised = response_json["data"][0]["revised_prompt"]
    assert isinstance(encrypted_revised, str)
    assert len(encrypted_revised) >= 64


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_generations_signature_verification(respx_mock):
    """Test that image generation responses can be verified via signature endpoint."""
    request_data = {
        "model": "dall-e-3",
        "prompt": "A dog playing in the park",
        "n": 1,
    }

    response_data = {
        "created": 1713833628,
        "data": [
            {
                "b64_json": "base64imagedata",
                "revised_prompt": "A happy dog playing fetch in a sunny park",
            }
        ],
    }

    route = respx_mock.post(VLLM_IMAGES_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    response_json = response.json()
    image_id = response_json["id"]

    # Verify we can fetch the signature
    signature_response = client.get(
        f"/v1/signature/{image_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert "text" in signature_data
    assert "signature" in signature_data
    assert signature_data["signature"].startswith("0x")
    assert "signing_address" in signature_data

    # Verify the signed text contains request:response hash format
    assert ":" in signature_data["text"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_generations_multiple_images(respx_mock):
    """Test image generation with multiple images (n > 1)."""
    plain_prompt = "Abstract art"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    request_data = {
        "model": "dall-e-3",
        "prompt": encrypted_prompt,
        "n": 2,
    }

    response_data = {
        "created": 1713833628,
        "data": [
            {"b64_json": "image1base64", "revised_prompt": "Abstract art piece 1"},
            {"b64_json": "image2base64", "revised_prompt": "Abstract art piece 2"},
        ],
    }

    route = respx_mock.post(VLLM_IMAGES_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_json = response.json()

    # Verify both images are encrypted
    assert len(response_json["data"]) == 2
    for item in response_json["data"]:
        assert len(item["b64_json"]) >= 64
        assert len(item["revised_prompt"]) >= 64


# ==================== Streaming with Multimodal Content Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_multimodal_streaming(respx_mock):
    """Test streaming chat completions with multimodal content."""
    multimodal_content = [
        {"type": "text", "text": "Describe what you see"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": True,
    }

    chat_id = "chatcmpl-stream-multimodal-123"

    async def yield_sse_response():
        chunks = [
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "I see"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": " a photo"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
            },
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=yield_sse_response(),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert route.called

    # Verify the request was decrypted correctly
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert isinstance(sent_data["messages"][0]["content"], list)


# ==================== Error Handling Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_images_generations_upstream_error(respx_mock):
    """Test image generation handles upstream errors correctly."""
    request_data = {
        "model": "dall-e-3",
        "prompt": "A test prompt",
    }

    route = respx_mock.post(VLLM_IMAGES_URL).mock(
        return_value=httpx.Response(500, json={"error": "Internal error"})
    )

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 500
    assert route.called


@pytest.mark.asyncio
async def test_images_generations_invalid_json():
    """Test image generation rejects invalid JSON."""
    response = client.post(
        "/v1/images/generations",
        content=b"not valid json",
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_images_generations_invalid_encryption_headers():
    """Test image generation validates encryption headers."""
    request_data = {
        "model": "dall-e-3",
        "prompt": "A test prompt",
    }

    response = client.post(
        "/v1/images/generations",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "invalid-algo",
            "X-Client-Pub-Key": "some-key",
        },
    )

    assert response.status_code == 400


# ==================== Hash Verification Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_multimodal_signature_verification(respx_mock):
    """Test signature verification for multimodal chat completions."""
    multimodal_content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-verify-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is an image."},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_json = response.json()

    # Fetch and verify signature
    signature_response = client.get(
        f"/v1/signature/{chat_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert signature_response.status_code == 200
    signature_data = signature_response.json()

    # Calculate expected request hash
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    # Calculate expected response hash (using actual encrypted response)
    encrypted_response_body = json.dumps(response_json, separators=(",", ":")).encode("utf-8")
    expected_response_hash = sha256(encrypted_response_body).hexdigest()

    # Verify the signed text matches
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
