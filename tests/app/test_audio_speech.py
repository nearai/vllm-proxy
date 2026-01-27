"""Tests for /v1/audio/speech endpoint (text-to-speech)."""

import httpx
import pytest
from fastapi.testclient import TestClient
import json
from hashlib import sha256

# Import and setup test environment before importing app
from tests.app.test_helpers import setup_test_environment, TEST_AUTH_HEADER

# Setup all mocks before importing app
setup_test_environment()

# Import real quote contexts (not needed for basic tests, but good for encryption)
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


# ==================== Basic Speech Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_basic_success(respx_mock):
    """Test basic speech generation returns binary audio."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"

    route = respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/mpeg"})
    )

    request_data = {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "X-Response-ID" in response.headers
    assert response.headers["X-Response-ID"].startswith("speech-")
    assert response.content == audio_data
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_missing_authorization(respx_mock):
    """Test speech without authorization header."""
    request_data = {
        "model": "tts-1",
        "input": "Hello",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
    )

    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_invalid_json(respx_mock):
    """Test speech with invalid JSON request."""
    response = client.post(
        "/v1/audio/speech",
        content=b"not valid json",
        headers={
            "Content-Type": "application/json",
            "Authorization": TEST_AUTH_HEADER,
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_with_response_format(respx_mock):
    """Test speech with different response formats."""
    audio_data = b"FLAC\x00\x00\x00\x22"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": "audio/flac"})
    )

    request_data = {
        "model": "tts-1-hd",
        "input": "Test audio",
        "voice": "echo",
        "response_format": "flac",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/flac"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_with_speed(respx_mock):
    """Test speech with custom speed parameter."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    request_data = {
        "model": "tts-1",
        "input": "Slow speech",
        "voice": "nova",
        "speed": 0.5,
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_multiple_voices(respx_mock):
    """Test speech with different voice options."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    for voice in voices:
        request_data = {
            "model": "tts-1",
            "input": f"Test with {voice}",
            "voice": voice,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_response_id_generation(respx_mock):
    """Test that response ID is correctly generated."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    request_data = {
        "model": "tts-1",
        "input": "Test ID",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    response_id = response.headers["X-Response-ID"]
    assert response_id.startswith("speech-")
    assert len(response_id) == len("speech-") + 24


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_upstream_error(respx_mock):
    """Test speech when upstream service returns error."""
    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(500, text="Internal server error")
    )

    request_data = {
        "model": "tts-1",
        "input": "Test error",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 500
    assert "Upstream service error" in response.text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_signature_caching(respx_mock):
    """Test that signatures are properly cached."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    request_data = {
        "model": "tts-1",
        "input": "Test signature",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    response_id = response.headers["X-Response-ID"]

    # Verify signature can be retrieved
    sig_response = client.get(
        f"/v1/signature/{response_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert sig_response.status_code == 200
    sig_data = sig_response.json()
    assert "text" in sig_data
    assert "signature" in sig_data
    assert "signing_address" in sig_data


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_large_audio_output(respx_mock):
    """Test speech with large audio output."""
    # Create 5MB of audio data
    large_audio = b"AUDIO" * (1024 * 1024)

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=large_audio, headers={"content-type": "audio/mpeg"})
    )

    request_data = {
        "model": "tts-1",
        "input": "Very long text " * 100,
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert len(response.content) == len(large_audio)


@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.parametrize("response_format,content_type", [
    ("mp3", "audio/mpeg"),
    ("opus", "audio/opus"),
    ("aac", "audio/aac"),
    ("flac", "audio/flac"),
    ("wav", "audio/wav"),
    ("pcm", "audio/pcm"),
])
async def test_audio_formats(respx_mock, response_format, content_type):
    """Test all supported audio formats."""
    audio_data = b"AUDIO_DATA"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data, headers={"content-type": content_type})
    )

    request_data = {
        "model": "tts-1",
        "input": "Format test",
        "voice": "alloy",
        "response_format": response_format,
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == content_type


@pytest.mark.asyncio
@pytest.mark.respx
async def test_default_content_type(respx_mock):
    """Test default content-type when not specified in response."""
    audio_data = b"AUDIO"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    request_data = {
        "model": "tts-1",
        "input": "Test",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    # Should default to audio/mpeg
    assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_correct_request_hash(respx_mock):
    """Test speech with correct request hash header."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"

    respx_mock.post(VLLM_SPEECH_URL).mock(
        return_value=httpx.Response(200, content=audio_data)
    )

    request_data = {
        "model": "tts-1",
        "input": "Test hash",
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
            "X-Request-Hash": correct_hash,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Response-ID"].startswith("speech-")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_speech_incorrect_request_hash(respx_mock):
    """Test speech with incorrect request hash returns 400."""
    request_data = {
        "model": "tts-1",
        "input": "Test hash",
        "voice": "alloy",
    }

    # Provide an incorrect hash
    incorrect_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Request-Hash": incorrect_hash,
        },
    )

    # Should reject with 400 due to hash mismatch
    assert response.status_code == 400
    assert "X-Request-Hash mismatch" in response.json()["detail"]


# ==================== Input Validation Tests ====================


@pytest.mark.asyncio
async def test_speech_missing_model():
    """Test speech request with missing model field."""
    request_data = {
        "input": "Test",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "missing required field" in response.json()["detail"].lower()
    assert "model" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_missing_input():
    """Test speech request with missing input field."""
    request_data = {
        "model": "tts-1",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "missing required field" in response.json()["detail"].lower()
    assert "input" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_missing_voice():
    """Test speech request with missing voice field."""
    request_data = {
        "model": "tts-1",
        "input": "Test",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "missing required field" in response.json()["detail"].lower()
    assert "voice" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_empty_input():
    """Test speech request with empty input."""
    request_data = {
        "model": "tts-1",
        "input": "",
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_invalid_voice():
    """Test speech request with invalid voice."""
    request_data = {
        "model": "tts-1",
        "input": "Test",
        "voice": "invalid_voice",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "invalid voice" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_invalid_response_format():
    """Test speech request with invalid response_format."""
    request_data = {
        "model": "tts-1",
        "input": "Test",
        "voice": "alloy",
        "response_format": "invalid_format",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "invalid response_format" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_invalid_speed():
    """Test speech request with invalid speed."""
    request_data = {
        "model": "tts-1",
        "input": "Test",
        "voice": "alloy",
        "speed": 5.0,  # Out of range [0.25, 4.0]
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "speed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_speech_input_exceeds_max_length():
    """Test speech request with input exceeding max length."""
    request_data = {
        "model": "tts-1",
        "input": "x" * 4097,  # Exceeds 4096 limit
        "voice": "alloy",
    }

    response = client.post(
        "/v1/audio/speech",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 400
    assert "exceeds maximum length" in response.json()["detail"].lower()
