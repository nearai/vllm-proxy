"""
Tests for end-to-end encryption on the image edits endpoint.

These tests verify that:
1. The image edits endpoint can decrypt encrypted prompts
2. The image edits response is properly encrypted (b64_json, revised_prompt)
3. Both ECDSA and Ed25519 encryption are supported
"""

import httpx
import pytest
from fastapi.testclient import TestClient
import json
from hashlib import sha256
from unittest.mock import patch

# Import and setup test environment before importing app
from tests.app.test_helpers import setup_test_environment, TEST_AUTH_HEADER

# Setup all mocks before importing app
setup_test_environment()

# Import real quote contexts BEFORE replacing with mock (needed for encryption)
import sys
import app.quote.quote as real_quote
from app.encryption.encryption import encrypt_data, decrypt_data

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
from app.api.v1.openai import VLLM_IMAGES_EDITS_URL

client = TestClient(app)

# Test image data (minimal PNG header + padding)
TEST_IMAGE = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def encrypt_content(content: str, signing_algo: str) -> str:
    """Helper to encrypt content using the server's public key."""
    if signing_algo == ECDSA:
        public_key = real_ecdsa_context.signing_public_key
    else:
        public_key = real_ed25519_context.signing_public_key

    encrypted_data = encrypt_data(content.encode("utf-8"), public_key, signing_algo)
    return encrypted_data.hex()


def decrypt_content(encrypted_hex: str, signing_algo: str) -> str:
    """Helper to decrypt content using the server's private key."""
    if signing_algo == ECDSA:
        context = real_ecdsa_context
    else:
        context = real_ed25519_context

    encrypted_data = bytes.fromhex(encrypted_hex)
    decrypted_data = decrypt_data(encrypted_data, context)
    return decrypted_data.decode("utf-8")


# ==================== Image Edits Endpoint Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_ecdsa(respx_mock):
    """Test encrypted image edit request with ECDSA."""
    # Encrypt the prompt
    plain_prompt = "Add a beautiful sunset to the background"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Mock response data
    response_id = "img-edit-encrypted-ecdsa-123"
    b64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    revised_prompt = "A beautiful sunset added to the background of the image"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": b64_image,
                "revised_prompt": revised_prompt,
            }
        ],
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response structure
    assert response_json["id"] == response_id
    assert "data" in response_json
    assert len(response_json["data"]) == 1

    # Verify b64_json is encrypted (hex string)
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert isinstance(encrypted_b64, str)
    assert len(encrypted_b64) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify revised_prompt is encrypted (hex string)
    encrypted_revised = response_json["data"][0]["revised_prompt"]
    assert isinstance(encrypted_revised, str)
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_revised)

    # Verify the prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    # For multipart requests, we need to parse the form data
    # The prompt should be in the form data as plain text
    assert plain_prompt.encode() in call_args.content

    # Verify we can decrypt the response fields back to original
    decrypted_b64 = decrypt_content(encrypted_b64, ECDSA)
    assert decrypted_b64 == b64_image

    decrypted_revised = decrypt_content(encrypted_revised, ECDSA)
    assert decrypted_revised == revised_prompt


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_ed25519(respx_mock):
    """Test encrypted image edit request with Ed25519."""
    # Encrypt the prompt
    plain_prompt = "Make the sky more vibrant"
    encrypted_prompt = encrypt_content(plain_prompt, ED25519)

    # Mock response data
    response_id = "img-edit-encrypted-ed25519-456"
    b64_image = "base64encodedimagedata"
    revised_prompt = "The sky has been made more vibrant with enhanced colors"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": b64_image,
                "revised_prompt": revised_prompt,
            }
        ],
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ED25519,
                "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
            },
        )

    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify b64_json is encrypted
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert isinstance(encrypted_b64, str)
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify revised_prompt is encrypted
    encrypted_revised = response_json["data"][0]["revised_prompt"]
    assert isinstance(encrypted_revised, str)
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_revised)

    # Verify prompt was decrypted before forwarding
    call_args = route.calls[0].request
    assert plain_prompt.encode() in call_args.content

    # Verify we can decrypt the response fields
    decrypted_b64 = decrypt_content(encrypted_b64, ED25519)
    assert decrypted_b64 == b64_image

    decrypted_revised = decrypt_content(encrypted_revised, ED25519)
    assert decrypted_revised == revised_prompt


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_multiple_images(respx_mock):
    """Test encrypted image edit request with multiple images (ECDSA)."""
    # Encrypt the prompt
    plain_prompt = "Combine these images with a gradient effect"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Mock response data
    response_id = "img-edit-multi-encrypted-789"
    b64_image = "combinedimagebase64data"
    revised_prompt = "Images combined with a smooth gradient effect"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": b64_image,
                "revised_prompt": revised_prompt,
            }
        ],
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Create multiple test images
    test_image_1 = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100
    test_image_2 = b"\x89PNG\r\n\x1a\n" + b"\x02" * 100

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("image1.png", test_image_1, "image/png")),
                ("image[]", ("image2.png", test_image_2, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    # Verify response
    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify encrypted response
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify decryption
    decrypted_b64 = decrypt_content(encrypted_b64, ECDSA)
    assert decrypted_b64 == b64_image


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_multiple_outputs(respx_mock):
    """Test encrypted image edit request with n>1 (multiple output images)."""
    plain_prompt = "Create variations of this image"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Mock response with multiple images
    response_id = "img-edit-multi-out-abc"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "firstimage64data",
                "revised_prompt": "First variation of the image",
            },
            {
                "b64_json": "secondimage64data",
                "revised_prompt": "Second variation of the image",
            },
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
                "n": "2",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify all output images are encrypted
    assert len(response_json["data"]) == 2
    for i, item in enumerate(response_json["data"]):
        encrypted_b64 = item["b64_json"]
        assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

        encrypted_revised = item["revised_prompt"]
        assert all(c in "0123456789abcdefABCDEF" for c in encrypted_revised)

        # Verify decryption
        decrypted_b64 = decrypt_content(encrypted_b64, ECDSA)
        assert decrypted_b64 == response_data["data"][i]["b64_json"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_image_edits_no_encryption(respx_mock):
    """Test image edit request without encryption headers (passthrough)."""
    plain_prompt = "Add some flowers to the garden"

    response_id = "img-edit-plain-123"
    b64_image = "plaintextbase64image"
    revised_prompt = "Flowers added to the garden"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": b64_image,
                "revised_prompt": revised_prompt,
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": plain_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response is NOT encrypted (plain text)
    assert response_json["data"][0]["b64_json"] == b64_image
    assert response_json["data"][0]["revised_prompt"] == revised_prompt


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_invalid_signing_algo(respx_mock):
    """Test that invalid signing algorithm is rejected."""
    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "Some prompt",
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", TEST_IMAGE, "image/png")),
        ],
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "invalid-algo",
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400
    assert "Invalid X-Signing-Algo" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_invalid_pub_key_format(respx_mock):
    """Test that invalid public key format is rejected."""
    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "Some prompt",
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", TEST_IMAGE, "image/png")),
        ],
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "not-a-valid-hex!@#$",
        },
    )

    assert response.status_code == 400
    assert "valid hex string" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_wrong_key_length_ecdsa(respx_mock):
    """Test that wrong key length for ECDSA is rejected."""
    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "Some prompt",
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", TEST_IMAGE, "image/png")),
        ],
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "aa" * 32,  # 32 bytes instead of 64
        },
    )

    assert response.status_code == 400
    assert "128 hex characters" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_wrong_key_length_ed25519(respx_mock):
    """Test that wrong key length for Ed25519 is rejected."""
    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "Some prompt",
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", TEST_IMAGE, "image/png")),
        ],
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": "bb" * 64,  # 64 bytes instead of 32
        },
    )

    assert response.status_code == 400
    assert "64 hex characters (32 bytes)" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_decryption_failure(respx_mock):
    """Test that invalid encrypted prompt causes decryption failure."""
    # Use invalid encrypted data (not properly encrypted)
    invalid_encrypted_prompt = "aa" * 100  # Random hex, not properly encrypted

    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": invalid_encrypted_prompt,
            "model": "gpt-image-1.5",
        },
        files=[
            ("image[]", ("test.png", TEST_IMAGE, "image/png")),
        ],
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400
    assert "Failed to decrypt" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_with_request_hash(respx_mock):
    """Test encrypted image edit with X-Request-Hash header."""
    plain_prompt = "Add a rainbow"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)
    custom_hash = "custom-image-edit-hash-123"

    response_id = "img-edit-hash-test"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "testimage",
                "revised_prompt": "Rainbow added",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
                "X-Request-Hash": custom_hash,
            },
        )

        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {custom_hash}"
        )

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_partial_headers(respx_mock):
    """Test that partial encryption headers means no encryption (passthrough)."""
    plain_prompt = "Add clouds"

    response_id = "img-edit-partial-123"
    b64_image = "plainbase64"
    response_data = {
        "created": 1677825464,
        "data": [{"b64_json": b64_image}],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        # Only provide X-Signing-Algo, not X-Client-Pub-Key
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": plain_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                # Missing X-Client-Pub-Key
            },
        )

    # Encryption is not enabled when only one header is provided
    # So it should pass through without encryption
    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response is NOT encrypted
    assert response_json["data"][0]["b64_json"] == b64_image


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_with_optional_params(respx_mock):
    """Test encrypted image edit with optional parameters (size, quality, etc.)."""
    plain_prompt = "Make it high quality"
    encrypted_prompt = encrypt_content(plain_prompt, ED25519)

    response_id = "img-edit-opts-xyz"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "hqimagedata",
                "revised_prompt": "High quality image generated",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
                "n": "1",
                "size": "1024x1024",
                "quality": "hd",
                "response_format": "b64_json",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ED25519,
                "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response is encrypted
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify decryption
    decrypted_b64 = decrypt_content(encrypted_b64, ED25519)
    assert decrypted_b64 == "hqimagedata"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_only_b64_json_no_revised_prompt(respx_mock):
    """Test encrypted image edit when response has b64_json but no revised_prompt."""
    plain_prompt = "Simple edit"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    response_id = "img-edit-simple-123"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "simplebase64",
                # No revised_prompt in response
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify b64_json is encrypted
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify decryption
    decrypted_b64 = decrypt_content(encrypted_b64, ECDSA)
    assert decrypted_b64 == "simplebase64"

    # Verify revised_prompt is not present
    assert "revised_prompt" not in response_json["data"][0]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_generates_id_if_missing(respx_mock):
    """Test that images edits endpoint generates ID if not in response."""
    plain_prompt = "Test prompt"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Response without ID
    response_data = {
        "created": 1677825464,
        "data": [{"b64_json": "testdata"}],
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    response_json = response.json()

    # Verify ID was generated with correct prefix
    assert "id" in response_json
    assert response_json["id"].startswith("img-")
    assert len(response_json["id"]) == 28  # "img-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_with_background_param(respx_mock):
    """Test encrypted image edit with background parameter."""
    plain_prompt = "Make background transparent"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    response_id = "img-edit-bg-123"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "transparentbgimage",
                "revised_prompt": "Image with transparent background",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
                "background": "transparent",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called

    # Verify background parameter was forwarded
    call_args = route.calls[0].request
    assert b"transparent" in call_args.content

    # Verify response is encrypted
    response_json = response.json()
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_image_edits_with_mask(respx_mock):
    """Test encrypted image edit with mask parameter."""
    plain_prompt = "Edit the masked area"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    response_id = "img-edit-mask-456"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "editedmaskarea",
                "revised_prompt": "Edited the masked area of the image",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Create a test mask image (PNG with transparency)
    mask_image = b"\x89PNG\r\n\x1a\n" + b"\xff" * 100

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": encrypted_prompt,
                "model": "gpt-image-1.5",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
                ("mask", ("mask.png", mask_image, "image/png")),
            ],
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called

    # Verify mask was forwarded (check that it's in the multipart request)
    call_args = route.calls[0].request
    assert b"mask" in call_args.content

    # Verify response is encrypted
    response_json = response.json()
    encrypted_b64 = response_json["data"][0]["b64_json"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_b64)

    # Verify we can decrypt
    decrypted_b64 = decrypt_content(encrypted_b64, ECDSA)
    assert decrypted_b64 == "editedmaskarea"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_image_edits_with_mask_no_encryption(respx_mock):
    """Test image edit with mask but without encryption."""
    plain_prompt = "Fill in the masked area"

    response_id = "img-edit-mask-plain-789"
    response_data = {
        "created": 1677825464,
        "data": [
            {
                "b64_json": "plainmaskresult",
            }
        ],
        "id": response_id,
    }

    route = respx_mock.post(VLLM_IMAGES_EDITS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    mask_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": plain_prompt,
                "model": "gpt-image-1.5",
                "background": "opaque",
            },
            files=[
                ("image[]", ("test.png", TEST_IMAGE, "image/png")),
                ("mask", ("mask.png", mask_image, "image/png")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response is NOT encrypted (plain text)
    assert response_json["data"][0]["b64_json"] == "plainmaskresult"
