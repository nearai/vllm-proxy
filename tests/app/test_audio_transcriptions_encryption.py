"""
Tests for end-to-end encryption on the audio transcriptions endpoint.

These tests verify that:
1. The audio transcriptions endpoint can decrypt encrypted prompts
2. The audio transcriptions response is properly encrypted (text field)
3. Both ECDSA and Ed25519 encryption are supported
"""

import httpx
import pytest
from fastapi.testclient import TestClient
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
from app.api.v1.openai import VLLM_TRANSCRIPTIONS_URL

client = TestClient(app)

# Test audio data (minimal WAV header + padding)
TEST_AUDIO = b"RIFF" + b"\x00" * 100


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


# ==================== Audio Transcriptions Endpoint Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_ecdsa(respx_mock):
    """Test encrypted audio transcription request with ECDSA."""
    # Encrypt the prompt
    plain_prompt = "This is a technical discussion about AI"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Mock response data
    response_id = "trans-encrypted-ecdsa-123"
    transcription_text = "Hello, this is a test transcription of the audio file."
    response_data = {
        "text": transcription_text,
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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

    # Verify text is encrypted (hex string)
    encrypted_text = response_json["text"]
    assert isinstance(encrypted_text, str)
    assert len(encrypted_text) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify the prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    # For multipart requests, the prompt should be in the form data as plain text
    assert plain_prompt.encode() in call_args.content

    # Verify we can decrypt the response text back to original
    decrypted_text = decrypt_content(encrypted_text, ECDSA)
    assert decrypted_text == transcription_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_ed25519(respx_mock):
    """Test encrypted audio transcription request with Ed25519."""
    # Encrypt the prompt
    plain_prompt = "French language audio clip"
    encrypted_prompt = encrypt_content(plain_prompt, ED25519)

    # Mock response data
    response_id = "trans-encrypted-ed25519-456"
    transcription_text = "Bonjour, comment allez-vous?"
    response_data = {
        "text": transcription_text,
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
                "language": "fr",
            },
            files=[
                ("file", ("test.mp3", TEST_AUDIO, "audio/mpeg")),
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

    # Verify text is encrypted
    encrypted_text = response_json["text"]
    assert isinstance(encrypted_text, str)
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify prompt was decrypted before forwarding
    call_args = route.calls[0].request
    assert plain_prompt.encode() in call_args.content

    # Verify we can decrypt the response text
    decrypted_text = decrypt_content(encrypted_text, ED25519)
    assert decrypted_text == transcription_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_no_prompt(respx_mock):
    """Test encrypted audio transcription without prompt (only response encrypted)."""
    # Mock response data
    response_id = "trans-no-prompt-789"
    transcription_text = "This is a transcription without a prompt."
    response_data = {
        "text": transcription_text,
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                # No prompt provided
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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

    # Verify text is encrypted even without prompt
    encrypted_text = response_json["text"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify decryption
    decrypted_text = decrypt_content(encrypted_text, ECDSA)
    assert decrypted_text == transcription_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_transcriptions_no_encryption(respx_mock):
    """Test audio transcription request without encryption headers (passthrough)."""
    plain_prompt = "Technical discussion"

    response_id = "trans-plain-123"
    transcription_text = "This is a plain text transcription."
    response_data = {
        "text": transcription_text,
        "id": response_id,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": plain_prompt,
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
            ],
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    assert route.called
    response_json = response.json()

    # Verify response is NOT encrypted (plain text)
    assert response_json["text"] == transcription_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_invalid_signing_algo(respx_mock):
    """Test that invalid signing algorithm is rejected."""
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
        },
        files=[
            ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_invalid_pub_key_format(respx_mock):
    """Test that invalid public key format is rejected."""
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
        },
        files=[
            ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_wrong_key_length_ecdsa(respx_mock):
    """Test that wrong key length for ECDSA is rejected."""
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
        },
        files=[
            ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_wrong_key_length_ed25519(respx_mock):
    """Test that wrong key length for Ed25519 is rejected."""
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
        },
        files=[
            ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_decryption_failure(respx_mock):
    """Test that invalid encrypted prompt causes decryption failure."""
    # Use invalid encrypted data (not properly encrypted)
    invalid_encrypted_prompt = "aa" * 100  # Random hex, not properly encrypted

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-large-v3",
            "prompt": invalid_encrypted_prompt,
        },
        files=[
            ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_with_request_hash(respx_mock):
    """Test encrypted audio transcription with X-Request-Hash header."""
    plain_prompt = "Medical terminology"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)
    custom_hash = "custom-transcription-hash-123"

    response_id = "trans-hash-test"
    response_data = {
        "text": "Medical transcription result",
        "id": response_id,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
async def test_encrypted_audio_transcriptions_partial_headers(respx_mock):
    """Test that partial encryption headers means no encryption (passthrough)."""
    plain_prompt = "Simple prompt"

    response_id = "trans-partial-123"
    transcription_text = "Plain transcription text"
    response_data = {
        "text": transcription_text,
        "id": response_id,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        # Only provide X-Signing-Algo, not X-Client-Pub-Key
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": plain_prompt,
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
    assert response_json["text"] == transcription_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_with_all_optional_params(respx_mock):
    """Test encrypted audio transcription with all optional parameters."""
    plain_prompt = "Spanish language news broadcast"
    encrypted_prompt = encrypt_content(plain_prompt, ED25519)

    response_id = "trans-opts-xyz"
    response_data = {
        "text": "Buenos dias, estas son las noticias.",
        "id": response_id,
        "language": "spanish",
        "duration": 5.2,
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
                "language": "es",
                "response_format": "verbose_json",
                "temperature": "0.0",
                "timestamp_granularities": "segment",
            },
            files=[
                ("file", ("test.mp3", TEST_AUDIO, "audio/mpeg")),
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

    # Verify text is encrypted
    encrypted_text = response_json["text"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify decryption
    decrypted_text = decrypt_content(encrypted_text, ED25519)
    assert decrypted_text == "Buenos dias, estas son las noticias."

    # Verify other fields are preserved (not encrypted)
    assert response_json["language"] == "spanish"
    assert response_json["duration"] == 5.2


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_generates_id_if_missing(respx_mock):
    """Test that audio transcriptions endpoint generates ID if not in response."""
    plain_prompt = "Test"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    # Response without ID
    response_data = {
        "text": "Generated ID test transcription",
    }

    route = respx_mock.post(VLLM_TRANSCRIPTIONS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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
    assert response_json["id"].startswith("trans-")
    assert len(response_json["id"]) == 30  # "trans-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_audio_transcriptions_verbose_json_response(respx_mock):
    """Test encrypted audio transcription with verbose_json response format."""
    plain_prompt = "Detailed transcription"
    encrypted_prompt = encrypt_content(plain_prompt, ECDSA)

    response_id = "trans-verbose-enc"
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

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-large-v3",
                "prompt": encrypted_prompt,
                "response_format": "verbose_json",
            },
            files=[
                ("file", ("test.wav", TEST_AUDIO, "audio/wav")),
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

    # Only the text field should be encrypted
    encrypted_text = response_json["text"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify we can decrypt
    decrypted_text = decrypt_content(encrypted_text, ECDSA)
    assert decrypted_text == "Hello world"

    # Other fields should be preserved unencrypted
    assert response_json["task"] == "transcribe"
    assert response_json["language"] == "english"
    assert response_json["duration"] == 2.5
    assert "segments" in response_json
