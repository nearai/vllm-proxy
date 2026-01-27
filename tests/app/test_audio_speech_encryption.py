"""Tests for /v1/audio/speech endpoint with end-to-end encryption."""

import base64
import json
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from app.encryption.encryption import encrypt_data
from app.main import app
from app.quote.quote import ECDSA, ED25519, ecdsa_context, ed25519_context

client = TestClient(app)

# Valid bearer token for tests
VALID_TOKEN = "test-token-12345"


@pytest.fixture
def auth_headers():
    """Return authorization headers."""
    return {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture
def ecdsa_public_key():
    """Get ECDSA public key for encryption."""
    return ecdsa_context.public_key_hex


@pytest.fixture
def ed25519_public_key():
    """Get Ed25519 public key for encryption."""
    return ed25519_context.public_key_hex


@pytest.fixture
def mock_vllm_speech_encrypted(monkeypatch, requests_mock):
    """Mock the vLLM speech endpoint for encrypted tests."""
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"

    requests_mock.post(
        "http://vllm:8000/v1/audio/speech",
        content=audio_data,
        headers={"content-type": "audio/mpeg"},
    )
    return audio_data


def encrypt_text_field(text: str, public_key: str, algo: str) -> str:
    """Encrypt text field using the same method as the endpoint."""
    encrypted_data = encrypt_data(text.encode("utf-8"), public_key, algo)
    return encrypted_data.hex()


class TestAudioSpeechEncryptionECDSA:
    """Test speech generation with ECDSA encryption."""

    def test_speech_encrypted_ecdsa(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test encrypted speech request and response with ECDSA."""
        # Encrypt the input text
        plaintext_input = "Hello world"
        encrypted_input = encrypt_text_field(plaintext_input, ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,  # Encrypted hex string
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
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

    def test_speech_encrypted_ecdsa_with_format(self, monkeypatch, requests_mock, auth_headers, ecdsa_public_key):
        """Test encrypted speech with response format."""
        audio_data = b"FLAC\x00\x00\x00\x22"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": "audio/flac"},
        )

        encrypted_input = encrypt_text_field("Test audio", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1-hd",
            "input": encrypted_input,
            "voice": "echo",
            "response_format": "flac",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["format"] == "flac"

    def test_speech_encrypted_ecdsa_with_speed(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test encrypted speech with speed parameter."""
        encrypted_input = encrypt_text_field("Slow speech", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "nova",
            "speed": 0.5,
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert isinstance(response_data["audio"], str)

    def test_speech_encrypted_ecdsa_custom_request_hash(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test encrypted speech with custom request hash."""
        custom_hash = "custom_hash_value"
        encrypted_input = encrypt_text_field("Test hash", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
            "X-Request-Hash": custom_hash,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200

    def test_speech_encrypted_ecdsa_invalid_public_key_format(self, mock_vllm_speech_encrypted, auth_headers):
        """Test encryption with invalid public key format."""
        request_data = {
            "model": "tts-1",
            "input": encrypt_text_field("Test", ecdsa_context.public_key_hex, ECDSA),
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "not_valid_hex_123",  # Invalid hex
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400

    def test_speech_encrypted_ecdsa_wrong_key_length(self, mock_vllm_speech_encrypted, auth_headers):
        """Test ECDSA encryption with wrong key length."""
        request_data = {
            "model": "tts-1",
            "input": encrypt_text_field("Test", ecdsa_context.public_key_hex, ECDSA),
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "abcd" * 10,  # Wrong length
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400

    def test_speech_encrypted_ecdsa_signature_generation(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test that signatures are generated for encrypted responses."""
        encrypted_input = encrypt_text_field("Signature test", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        response_id = response_data["id"]

        # Verify signature is cached
        sig_response = client.get(
            f"/v1/signature/{response_id}",
            headers=auth_headers,
        )

        assert sig_response.status_code == 200
        sig_data = sig_response.json()
        assert "signature_ecdsa" in sig_data
        assert sig_data["signature_ecdsa"]


class TestAudioSpeechEncryptionEd25519:
    """Test speech generation with Ed25519 encryption."""

    def test_speech_encrypted_ed25519(self, mock_vllm_speech_encrypted, auth_headers, ed25519_public_key):
        """Test encrypted speech request and response with Ed25519."""
        encrypted_input = encrypt_text_field("Hello world", ed25519_public_key, ED25519)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": ed25519_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
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

    def test_speech_encrypted_ed25519_with_format(self, monkeypatch, requests_mock, auth_headers, ed25519_public_key):
        """Test encrypted speech with Ed25519 and response format."""
        audio_data = b"OPUS_DATA"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": "audio/opus"},
        )

        encrypted_input = encrypt_text_field("Test", ed25519_public_key, ED25519)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "echo",
            "response_format": "opus",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": ed25519_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["format"] == "opus"

    def test_speech_encrypted_ed25519_invalid_key_length(self, mock_vllm_speech_encrypted, auth_headers):
        """Test Ed25519 encryption with wrong key length."""
        request_data = {
            "model": "tts-1",
            "input": encrypt_text_field("Test", ed25519_context.public_key_hex, ED25519),
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": "abcd" * 20,  # Wrong length for Ed25519 (needs 64 hex chars)
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400

    def test_speech_encrypted_ed25519_signature_generation(self, mock_vllm_speech_encrypted, auth_headers, ed25519_public_key):
        """Test that signatures are generated for Ed25519 encrypted responses."""
        encrypted_input = encrypt_text_field("Sig test", ed25519_public_key, ED25519)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": ed25519_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        response_id = response_data["id"]

        sig_response = client.get(
            f"/v1/signature/{response_id}",
            headers=auth_headers,
        )

        assert sig_response.status_code == 200
        sig_data = sig_response.json()
        assert "signature_ed25519" in sig_data
        assert sig_data["signature_ed25519"]


class TestAudioSpeechEncryptionValidation:
    """Test encryption header validation."""

    def test_missing_signing_algo(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test encryption with only public key (missing algo)."""
        request_data = {
            "model": "tts-1",
            "input": "plaintext",
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Client-Pub-Key": ecdsa_public_key,
            # Missing X-Signing-Algo
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        # Should treat as unencrypted and succeed
        assert response.status_code == 200

    def test_missing_public_key(self, mock_vllm_speech_encrypted, auth_headers):
        """Test encryption with only algo (missing public key)."""
        request_data = {
            "model": "tts-1",
            "input": "plaintext",
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            # Missing X-Client-Pub-Key
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        # Should treat as unencrypted and succeed
        assert response.status_code == 200

    def test_invalid_signing_algo(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test invalid signing algorithm."""
        request_data = {
            "model": "tts-1",
            "input": "test",
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": "rsa",  # Invalid algo
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400

    def test_empty_public_key(self, mock_vllm_speech_encrypted, auth_headers):
        """Test with empty public key."""
        request_data = {
            "model": "tts-1",
            "input": "test",
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400


class TestAudioSpeechEncryptionDecryption:
    """Test input text decryption."""

    def test_decryption_failure(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test handling of failed input decryption."""
        # Use invalid encrypted data (not actually encrypted)
        request_data = {
            "model": "tts-1",
            "input": "not_valid_encrypted_hex_data",
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 400

    def test_encrypted_input_decrypted_before_forward(self, monkeypatch, requests_mock, auth_headers, ecdsa_public_key):
        """Test that encrypted input is decrypted before forwarding to upstream."""
        plaintext_input = "Decrypted text"
        encrypted_input = encrypt_text_field(plaintext_input, ecdsa_public_key, ECDSA)

        audio_data = b"AUDIO_DATA"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
        )

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200

        # Check that the upstream received the decrypted input
        last_request = requests_mock.last_request
        upstream_data = json.loads(last_request.text)
        assert upstream_data["input"] == plaintext_input


class TestAudioSpeechEncryptionResponseEncoding:
    """Test audio response encryption and encoding."""

    def test_audio_base64_encoding(self, mock_vllm_speech_encrypted, auth_headers, ecdsa_public_key):
        """Test that audio is properly base64 encoded before encryption."""
        encrypted_input = encrypt_text_field("Test", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()

        # Audio field should be hex string representing encrypted base64
        audio_hex = response_data["audio"]
        # Verify it's valid hex
        audio_bytes = bytes.fromhex(audio_hex)
        assert len(audio_bytes) > 0

    def test_large_audio_encryption(self, monkeypatch, requests_mock, auth_headers, ecdsa_public_key):
        """Test encryption of large audio data."""
        # Create 10MB of audio data
        large_audio = b"AUDIO" * (2 * 1024 * 1024)  # 10MB
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=large_audio,
            headers={"content-type": "audio/mpeg"},
        )

        encrypted_input = encrypt_text_field("Large audio", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert "audio" in response_data


class TestAudioSpeechEncryptionMultipleFormats:
    """Test encryption with different audio formats."""

    @pytest.mark.parametrize("response_format", [
        "mp3",
        "opus",
        "aac",
        "flac",
        "wav",
        "pcm",
    ])
    def test_encrypted_with_different_formats(self, monkeypatch, requests_mock, auth_headers, ecdsa_public_key, response_format):
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

        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": content_type_map.get(response_format, "audio/mpeg")},
        )

        encrypted_input = encrypt_text_field("Format test", ecdsa_public_key, ECDSA)

        request_data = {
            "model": "tts-1",
            "input": encrypted_input,
            "voice": "alloy",
            "response_format": response_format,
        }

        headers = {
            **auth_headers,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": ecdsa_public_key,
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["format"] == response_format
