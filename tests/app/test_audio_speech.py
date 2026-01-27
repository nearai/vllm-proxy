"""Tests for /v1/audio/speech endpoint (text-to-speech)."""

import json
import uuid
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# Valid bearer token for tests
VALID_TOKEN = "test-token-12345"


@pytest.fixture
def mock_vllm_speech(monkeypatch, requests_mock):
    """Mock the vLLM speech endpoint."""
    # Create mock MP3 audio data (minimal valid MP3 frame)
    audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x00"  # Minimal MP3 header

    requests_mock.post(
        "http://vllm:8000/v1/audio/speech",
        content=audio_data,
        headers={"content-type": "audio/mpeg"},
    )
    return audio_data


@pytest.fixture
def auth_headers():
    """Return authorization headers."""
    return {"Authorization": f"Bearer {VALID_TOKEN}"}


class TestAudioSpeechBasic:
    """Test basic speech generation without encryption."""

    def test_speech_basic_success(self, mock_vllm_speech, auth_headers):
        """Test basic speech generation returns binary audio."""
        request_data = {
            "model": "tts-1",
            "input": "Hello world",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert "X-Response-ID" in response.headers
        assert response.headers["X-Response-ID"].startswith("speech-")
        assert response.content == mock_vllm_speech

    def test_speech_with_response_format(self, monkeypatch, requests_mock, auth_headers):
        """Test speech with different response formats."""
        audio_data = b"FLAC\x00\x00\x00\x22"  # Minimal FLAC header
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": "audio/flac"},
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
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/flac"

    def test_speech_with_speed(self, monkeypatch, requests_mock, auth_headers):
        """Test speech with custom speed parameter."""
        audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
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
            headers=auth_headers,
        )

        assert response.status_code == 200

    def test_speech_multiple_voices(self, monkeypatch, requests_mock, auth_headers):
        """Test speech with different voice options."""
        audio_data = b"ID3\x04\x00\x00\x00\x00\x00\x00"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
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
                headers=auth_headers,
            )

            assert response.status_code == 200

    def test_speech_response_id_generation(self, mock_vllm_speech, auth_headers):
        """Test that response ID is correctly generated."""
        request_data = {
            "model": "tts-1",
            "input": "Test ID",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_id = response.headers["X-Response-ID"]
        assert response_id.startswith("speech-")
        assert len(response_id) == len("speech-") + 24

    def test_speech_custom_request_hash(self, mock_vllm_speech, auth_headers):
        """Test speech with custom request hash header."""
        custom_hash = "custom_hash_value_12345"
        request_data = {
            "model": "tts-1",
            "input": "Test hash",
            "voice": "alloy",
        }

        headers = {**auth_headers, "X-Request-Hash": custom_hash}
        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200

    def test_speech_missing_authorization(self, mock_vllm_speech):
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

    def test_speech_invalid_json(self, auth_headers):
        """Test speech with invalid JSON request."""
        response = client.post(
            "/v1/audio/speech",
            content=b"not valid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == 400

    def test_speech_upstream_error(self, monkeypatch, requests_mock, auth_headers):
        """Test speech when upstream service returns error."""
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            status_code=500,
            text="Internal server error",
        )

        request_data = {
            "model": "tts-1",
            "input": "Test error",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 500
        assert "Upstream service error" in response.text

    def test_speech_large_audio_output(self, monkeypatch, requests_mock, auth_headers):
        """Test speech with large audio output."""
        # Create 5MB of audio data
        large_audio = b"AUDIO" * (1024 * 1024)  # 5MB
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=large_audio,
            headers={"content-type": "audio/mpeg"},
        )

        request_data = {
            "model": "tts-1",
            "input": "Very long text " * 100,
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert len(response.content) == len(large_audio)


class TestAudioSpeechSignature:
    """Test signature generation and verification."""

    def test_signature_caching(self, mock_vllm_speech, auth_headers):
        """Test that signatures are properly cached."""
        request_data = {
            "model": "tts-1",
            "input": "Test signature",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_id = response.headers["X-Response-ID"]

        # Verify signature can be retrieved
        sig_response = client.get(
            f"/v1/signature/{response_id}",
            headers=auth_headers,
        )

        assert sig_response.status_code == 200
        sig_data = sig_response.json()
        assert "text" in sig_data
        assert "signature_ecdsa" in sig_data
        assert "signature_ed25519" in sig_data

    def test_request_hash_in_signature(self, mock_vllm_speech, auth_headers):
        """Test that request hash is included in signature."""
        request_data = {
            "model": "tts-1",
            "input": "Hash test",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        response_id = response.headers["X-Response-ID"]

        sig_response = client.get(
            f"/v1/signature/{response_id}",
            headers=auth_headers,
        )

        assert sig_response.status_code == 200
        sig_data = sig_response.json()
        # Signature text should contain both request and response hashes
        assert ":" in sig_data["text"]


class TestAudioSpeechFormats:
    """Test different audio format outputs."""

    @pytest.mark.parametrize("response_format,content_type", [
        ("mp3", "audio/mpeg"),
        ("opus", "audio/opus"),
        ("aac", "audio/aac"),
        ("flac", "audio/flac"),
        ("wav", "audio/wav"),
        ("pcm", "audio/pcm"),
    ])
    def test_audio_formats(self, monkeypatch, requests_mock, auth_headers, response_format, content_type):
        """Test all supported audio formats."""
        audio_data = b"AUDIO_DATA"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": content_type},
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
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == content_type


class TestAudioSpeechContentType:
    """Test content-type handling."""

    def test_default_content_type(self, monkeypatch, requests_mock, auth_headers):
        """Test default content-type when not specified in response."""
        audio_data = b"AUDIO"
        # Don't set content-type header
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
        )

        request_data = {
            "model": "tts-1",
            "input": "Test",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        # Should default to audio/mpeg
        assert response.headers["content-type"] == "audio/mpeg"

    def test_preserve_upstream_content_type(self, monkeypatch, requests_mock, auth_headers):
        """Test that upstream content-type is preserved."""
        audio_data = b"OPUS_DATA"
        requests_mock.post(
            "http://vllm:8000/v1/audio/speech",
            content=audio_data,
            headers={"content-type": "audio/opus; charset=utf-8"},
        )

        request_data = {
            "model": "tts-1",
            "input": "Test",
            "voice": "alloy",
        }

        response = client.post(
            "/v1/audio/speech",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "audio/opus" in response.headers["content-type"]
