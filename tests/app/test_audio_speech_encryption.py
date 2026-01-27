"""Tests for /v1/audio/speech endpoint with end-to-end encryption."""

import base64
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
mock_quote_module = __import__("tests.app.mock_quote", fromlist=[""])
sys.modules["app.quote.quote"] = mock_quote_module

# Replace the mock contexts with real contexts so decryption works
mock_quote_module.ecdsa_context = real_ecdsa_context
mock_quote_module.ed25519_context = real_ed25519_context
mock_quote_module.ecdsa_quote = real_ecdsa_context
mock_quote_module.ed25519_quote = real_ed25519_context

# Now we can safely import app code
from app.main import app
from app.api.v1.openai import VLLM_SPEECH_URL

client = TestClient(app)


def encrypt_text_field(text: str, public_key: str, algo: str) -> str:
    """Encrypt text field using the same method as the endpoint."""
    encrypted_data = encrypt_data(text.encode("utf-8"), public_key, algo)
    return encrypted_data.hex()


# ==================== ECDSA Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ecdsa(respx_mock):
    """Test encrypted speech request and response with ECDSA."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/mpeg"})
    )

    # Encrypt the input text
    plaintext_input = "Hello world"
    encrypted_input = encrypt_text_field(plaintext_input, real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,  # Encrypted hex string
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Should be JSON response with encrypted audio
    response_data = response.json()
    assert "id" in response_data
    assert response_data["id"].startswith("speech-")
    assert "audio" in response_data
    assert response_data["format"] == "mp3"

    # Audio field should be hex string (encrypted base64)
    assert isinstance(response_data["audio"], str)
    # Should be valid hex
    int(response_data["audio"], 16)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ecdsa_with_format(respx_mock):
    """Test encrypted speech with response format."""
    audio_data = b"FLAC\x00\x00\x00\x22"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/flac"})
    )

    encrypted_input = encrypt_text_field("Test audio", real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1-hd",
        "input": encrypted_input,
        "voice": "echo",
        "response_format": "flac",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["format"] == "flac"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ecdsa_signature(respx_mock):
    """Test that signatures are generated for encrypted responses."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    encrypted_input = encrypt_text_field("Signature test", real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    response_id = response_data["id"]

    # Verify signature is cached
    sig_response = client.get(
        f"/v1/signature/{response_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert sig_response.status_code == 200
    sig_data = sig_response.json()
    assert "signature" in sig_data
    assert sig_data["signature"]


# ==================== Ed25519 Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ed25519(respx_mock):
    """Test encrypted speech request and response with Ed25519."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/mpeg"})
    )

    encrypted_input = encrypt_text_field("Hello world", real_ed25519_context.signing_public_key, ED25519)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    response_data = response.json()
    assert "id" in response_data
    assert response_data["id"].startswith("speech-")
    assert "audio" in response_data
    assert isinstance(response_data["audio"], str)
    # Verify it's valid hex
    int(response_data["audio"], 16)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ed25519_with_format(respx_mock):
    """Test encrypted speech with Ed25519 and response format."""
    audio_data = b"OPUS_DATA"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/opus"})
    )

    encrypted_input = encrypt_text_field("Test", real_ed25519_context.signing_public_key, ED25519)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "echo",
        "response_format": "opus",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["format"] == "opus"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ed25519_signature(respx_mock):
    """Test that signatures are generated for Ed25519 encrypted responses."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    encrypted_input = encrypt_text_field("Sig test", real_ed25519_context.signing_public_key, ED25519)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    response_id = response_data["id"]

    sig_response = client.get(
        f"/v1/signature/{response_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert sig_response.status_code == 200
    sig_data = sig_response.json()
    assert "signature" in sig_data
    assert sig_data["signature"]


# ==================== Encryption Validation Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_invalid_signing_algo(respx_mock):
    """Test invalid signing algorithm."""
    request_data = {
        "model": "tts-1",
        "input": "test",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "rsa",  # Invalid algo
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_invalid_public_key_format(respx_mock):
    """Test encryption with invalid public key format."""
    request_data = {
        "model": "tts-1",
        "input": encrypt_text_field("Test", real_ecdsa_context.signing_public_key, ECDSA),
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "not_valid_hex_123",  # Invalid hex
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ecdsa_wrong_key_length(respx_mock):
    """Test ECDSA encryption with wrong key length."""
    request_data = {
        "model": "tts-1",
        "input": encrypt_text_field("Test", real_ecdsa_context.signing_public_key, ECDSA),
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "abcd" * 10,  # Wrong length
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_encrypted_ed25519_invalid_key_length(respx_mock):
    """Test Ed25519 encryption with wrong key length."""
    request_data = {
        "model": "tts-1",
        "input": encrypt_text_field("Test", real_ed25519_context.signing_public_key, ED25519),
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": "abcd" * 20,  # Wrong length for Ed25519
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_speech_decryption_failure():
    """Test handling of failed input decryption."""
    # Use invalid encrypted data (not actually encrypted)
    request_data = {
        "model": "tts-1",
        "input": "not_valid_encrypted_hex_data",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400


# ==================== Encryption with Different Formats ====================


@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.parametrize("response_format", [
    "mp3",
    "opus",
    "aac",
    "flac",
    "wav",
    "pcm",
])
async def test_encrypted_with_different_formats(respx_mock, response_format):
    """Test encrypted response with different audio formats."""
    audio_data = b"AUDIO_DATA_FORMAT"
    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(
            200,
            content=audio_data,
            headers={"content-type": content_type_map.get(response_format, "audio/mpeg")}
        )
    )

    encrypted_input = encrypt_text_field("Format test", real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
        "response_format": response_format,
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["format"] == response_format


# ==================== Request Hash Verification Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_correct_request_hash(respx_mock):
    """Test encrypted request with correct request hash header."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    encrypted_input = encrypt_text_field("Test", real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    # Calculate the correct hash for the request_data
    request_body_bytes = json.dumps(request_data).encode("utf-8")
    correct_hash = sha256(request_body_bytes).hexdigest()

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            "X-Request-Hash": correct_hash,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "id" in response_data
    assert response_data["id"].startswith("speech-")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_incorrect_request_hash(respx_mock):
    """Test encrypted request with incorrect request hash returns 400."""
    encrypted_input = encrypt_text_field("Test", real_ecdsa_context.signing_public_key, ECDSA)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    # Provide an incorrect hash
    incorrect_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            "X-Request-Hash": incorrect_hash,
        },
    )

    # Should reject with 400 due to hash mismatch
    assert response.status_code == 400
    assert "X-Request-Hash mismatch" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_ed25519_correct_request_hash(respx_mock):
    """Test encrypted request with Ed25519 and correct request hash header."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    encrypted_input = encrypt_text_field("Test", real_ed25519_context.signing_public_key, ED25519)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    # Calculate the correct hash for the request_data
    request_body_bytes = json.dumps(request_data).encode("utf-8")
    correct_hash = sha256(request_body_bytes).hexdigest()

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
            "X-Request-Hash": correct_hash,
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    assert "id" in response_data
    assert response_data["id"].startswith("speech-")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_ed25519_incorrect_request_hash(respx_mock):
    """Test encrypted request with Ed25519 and incorrect request hash returns 400."""
    encrypted_input = encrypt_text_field("Test", real_ed25519_context.signing_public_key, ED25519)

    request_data = {
        "model": "tts-1",
        "input": encrypted_input,
        "voice": "alloy",
    }

    # Provide an incorrect hash
    incorrect_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": real_ed25519_context.signing_public_key,
            "X-Request-Hash": incorrect_hash,
        },
    )

    # Should reject with 400 due to hash mismatch
    assert response.status_code == 400
    assert "X-Request-Hash mismatch" in response.json()["detail"]
