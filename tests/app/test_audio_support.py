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
from app.api.v1.openai import VLLM_URL

client = TestClient(app)


def encrypt_content(content: str, signing_algo: str) -> str:
    """Helper to encrypt content using the server's public key."""
    if signing_algo == ECDSA:
        public_key = real_ecdsa_context.signing_public_key
    else:
        public_key = real_ed25519_context.signing_public_key

    encrypted_data = encrypt_data(content.encode("utf-8"), public_key, signing_algo)
    return encrypted_data.hex()


# ==================== Audio Input Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_audio_url_no_encryption(respx_mock):
    """Test chat completions with audio URL input without encryption."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the content of this audio?"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/audio.wav"},
                    },
                ],
            }
        ],
        "stream": False,
    }

    # Mock response
    chat_id = "chatcmpl-audio-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The audio contains someone saying hello.",
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
    assert response_json["choices"][0]["message"]["content"] == "The audio contains someone saying hello."


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_base64_audio_no_encryption(respx_mock):
    """Test chat completions with base64 audio input without encryption."""
    # Simulated base64 audio data (WAV header + minimal data)
    base64_audio = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"},
                    },
                ],
            }
        ],
        "stream": False,
    }

    chat_id = "chatcmpl-base64-audio-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a silent audio clip."},
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

    # Verify the request was forwarded with the audio content intact
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"][0]["type"] == "text"
    assert sent_data["messages"][0]["content"][1]["type"] == "audio_url"


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_multimodal_audio_encrypted(respx_mock):
    """Test chat completions with encrypted multimodal content (text + audio)."""
    # Build multimodal content
    multimodal_content = [
        {"type": "text", "text": "What is said in this audio?"},
        {"type": "audio_url", "audio_url": {"url": "https://example.com/speech.mp3"}},
    ]

    # Serialize to JSON and encrypt
    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-multimodal-audio-enc-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "The audio says: Hello world!"},
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
    assert sent_data["messages"][0]["content"][0]["text"] == "What is said in this audio?"
    assert sent_data["messages"][0]["content"][1]["type"] == "audio_url"

    # Response content should be encrypted
    response_json = response.json()
    encrypted_response_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(encrypted_response_content, str)
    assert len(encrypted_response_content) >= 64  # Should be encrypted hex
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_response_content)


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_multimodal_audio_encrypted_ed25519(respx_mock):
    """Test chat completions with encrypted multimodal content using Ed25519."""
    multimodal_content = [
        {"type": "text", "text": "Analyze this audio recording"},
        {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAA="},
        },
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ED25519)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-audio-ed25519-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
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
    assert sent_data["messages"][0]["content"][0]["text"] == "Analyze this audio recording"


# ==================== Audio Output Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_audio_output_no_encryption(respx_mock):
    """Test chat completions with audio output (Qwen3-Omni) without encryption."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Say hello in audio"}],
        "modalities": ["audio"],
        "stream": False,
    }

    # Mock response with audio output
    chat_id = "chatcmpl-audio-output-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "index": 0,
                "finish_reason": "stop",
            },
            {
                "message": {
                    "role": "assistant",
                    "audio": {
                        "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARA==",
                        "format": "wav",
                    },
                },
                "index": 1,
                "finish_reason": "stop",
            },
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
    
    # Verify both text and audio choices are present
    assert len(response_json["choices"]) == 2
    assert response_json["choices"][0]["message"]["content"] == "Hello!"
    assert response_json["choices"][1]["message"]["audio"]["data"] == "UklGRiQAAABXQVZFZm10IBAAAAABAAEARA=="


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_audio_output_with_encryption(respx_mock):
    """Test chat completions with audio output encrypted."""
    plain_content = "Say hello"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "modalities": ["audio"],
        "stream": False,
    }

    # Mock response with audio output
    chat_id = "chatcmpl-audio-output-enc-123"
    audio_base64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARA=="
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello there!",
                },
                "index": 0,
                "finish_reason": "stop",
            },
            {
                "message": {
                    "role": "assistant",
                    "audio": {
                        "data": audio_base64,
                        "format": "wav",
                    },
                },
                "index": 1,
                "finish_reason": "stop",
            },
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

    # Verify prompt was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"] == plain_content

    response_json = response.json()
    
    # Verify text content is encrypted (hex string)
    encrypted_text_content = response_json["choices"][0]["message"]["content"]
    assert isinstance(encrypted_text_content, str)
    assert len(encrypted_text_content) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text_content)

    # Verify audio.data is encrypted (hex string)
    encrypted_audio_data = response_json["choices"][1]["message"]["audio"]["data"]
    assert isinstance(encrypted_audio_data, str)
    assert len(encrypted_audio_data) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_audio_data)
    # Audio data should be different from original (encrypted)
    assert encrypted_audio_data != audio_base64


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_text_only_modality(respx_mock):
    """Test chat completions with text-only modality (no audio output)."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Describe vLLM briefly."}],
        "modalities": ["text"],
        "stream": False,
    }

    chat_id = "chatcmpl-text-only-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "vLLM is a fast inference engine.",
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

    # Verify modalities is passed through to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["modalities"] == ["text"]

    response_json = response.json()
    assert len(response_json["choices"]) == 1
    assert response_json["choices"][0]["message"]["content"] == "vLLM is a fast inference engine."


# ==================== Streaming Audio Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_streaming_audio_no_encryption(respx_mock):
    """Test streaming chat completions with audio modality indicator."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Say hello"}],
        "modalities": ["audio"],
        "stream": True,
    }

    chat_id = "chatcmpl-stream-audio-123"

    async def yield_sse_response():
        chunks = [
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "audio",
                "choices": [{"delta": {"content": "UklGRiQAAAB="}, "index": 0}],
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
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 200
    assert route.called

    # Parse streaming response
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunk = json.loads(data)
                chunks.append(chunk)

    assert len(chunks) > 0
    
    # Verify modality field is preserved
    modalities_found = [c.get("modality") for c in chunks if "modality" in c]
    assert "text" in modalities_found
    assert "audio" in modalities_found


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_streaming_audio_with_encryption(respx_mock):
    """Test streaming audio with encryption enabled."""
    plain_content = "Generate speech"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "modalities": ["audio"],
        "stream": True,
    }

    chat_id = "chatcmpl-stream-audio-enc-123"

    async def yield_sse_response():
        chunks = [
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"delta": {"content": "Speech"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "audio",
                "choices": [{"delta": {"content": "AudioChunkData"}, "index": 0}],
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

    # Verify the request was decrypted
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["messages"][0]["content"] == plain_content

    # Parse streaming response
    chunks = []
    content = response.content.decode()
    for line in content.split("\n"):
        if line.startswith("data: "):
            data = line.replace("data: ", "").strip()
            if data and data != "[DONE]":
                chunk = json.loads(data)
                chunks.append(chunk)

    # Verify content in chunks is encrypted
    for chunk in chunks:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            choice = chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                delta_content = choice["delta"]["content"]
                if delta_content:
                    # Content should be encrypted (hex string)
                    assert isinstance(delta_content, str)
                    assert len(delta_content) >= 64
                    assert all(c in "0123456789abcdefABCDEF" for c in delta_content)


# ==================== Signature Verification Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_input_signature_verification(respx_mock):
    """Test signature verification for audio input chat completions."""
    multimodal_content = [
        {"type": "text", "text": "Transcribe this"},
        {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}},
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-audio-sig-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Transcription result."},
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
    assert signature_data["signature"].startswith("0x")


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_output_signature_verification(respx_mock):
    """Test signature verification for audio output responses."""
    plain_content = "Say hello"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "modalities": ["audio"],
        "stream": False,
    }

    chat_id = "chatcmpl-audio-out-sig-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "index": 0,
                "finish_reason": "stop",
            },
            {
                "message": {
                    "role": "assistant",
                    "audio": {"data": "AudioBase64Data", "format": "wav"},
                },
                "index": 1,
                "finish_reason": "stop",
            },
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

    # Verify audio data is encrypted
    assert len(response_json["choices"]) == 2
    encrypted_audio = response_json["choices"][1]["message"]["audio"]["data"]
    assert encrypted_audio != "AudioBase64Data"  # Should be encrypted

    # Fetch signature
    signature_response = client.get(
        f"/v1/signature/{chat_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert signature_response.status_code == 200
    signature_data = signature_response.json()

    # Calculate expected hashes
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()

    encrypted_response_body = json.dumps(response_json, separators=(",", ":")).encode("utf-8")
    expected_response_hash = sha256(encrypted_response_body).hexdigest()

    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"


# ==================== Extra Body Parameters Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_with_sampling_params_list(respx_mock):
    """Test that extra_body.sampling_params_list is passed through to vLLM."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "modalities": ["audio"],
        "extra_body": {
            "sampling_params_list": [
                {"temperature": 0.4, "top_p": 0.9, "max_tokens": 16384},
                {"temperature": 0.9, "top_k": 50, "max_tokens": 4096},
                {"temperature": 0.0, "max_tokens": 65536},
            ]
        },
        "stream": False,
    }

    chat_id = "chatcmpl-sampling-params-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi!"},
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

    # Verify extra_body is passed through
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert "extra_body" in sent_data
    assert "sampling_params_list" in sent_data["extra_body"]
    assert len(sent_data["extra_body"]["sampling_params_list"]) == 3


# ==================== Mixed Modalities Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_chat_completions_mixed_input_audio_and_image(respx_mock):
    """Test chat completions with both audio and image input (mixed modalities)."""
    multimodal_content = [
        {"type": "text", "text": "Compare the audio with the image"},
        {"type": "audio_url", "audio_url": {"url": "https://example.com/speech.wav"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    content_json = json.dumps(multimodal_content)
    encrypted_content = encrypt_content(content_json, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "stream": False,
    }

    chat_id = "chatcmpl-mixed-modalities-123"
    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": 1677825464,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "choices": [
            {
                "message": {"role": "assistant", "content": "The audio describes what's in the image."},
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

    # Verify all modalities were decrypted correctly
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    content = sent_data["messages"][0]["content"]
    
    assert isinstance(content, list)
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "audio_url"
    assert content[2]["type"] == "image_url"


# ==================== Error Handling Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_audio_upstream_error(respx_mock):
    """Test audio request handles upstream errors correctly."""
    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "https://example.com/bad.wav"}},
                ],
            }
        ],
        "stream": False,
    }

    route = respx_mock.post(VLLM_URL).mock(
        return_value=httpx.Response(500, json={"error": "Internal error"})
    )

    response = client.post(
        "/v1/chat/completions",
        json=request_data,
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert response.status_code == 500
    assert route.called


@pytest.mark.asyncio
@pytest.mark.respx
async def test_streaming_audio_signature_verification(respx_mock):
    """Test signature for streaming audio responses."""
    plain_content = "Speak"
    encrypted_content = encrypt_content(plain_content, ECDSA)

    request_data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": encrypted_content}],
        "modalities": ["audio"],
        "stream": True,
    }

    chat_id = "chatcmpl-stream-audio-sig-123"

    async def yield_sse_response():
        chunks = [
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            },
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "modality": "audio",
                "choices": [{"delta": {"content": "AudioData"}, "index": 0}],
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

    # Calculate expected hashes
    encrypted_request_body = json.dumps(request_data, separators=(",", ":")).encode("utf-8")
    expected_request_hash = sha256(encrypted_request_body).hexdigest()
    expected_response_hash = sha256(response.content).hexdigest()

    # Fetch signature
    signature_response = client.get(
        f"/v1/signature/{chat_id}",
        headers={"Authorization": TEST_AUTH_HEADER},
    )

    assert signature_response.status_code == 200
    signature_data = signature_response.json()
    assert signature_data["text"] == f"{expected_request_hash}:{expected_response_hash}"
