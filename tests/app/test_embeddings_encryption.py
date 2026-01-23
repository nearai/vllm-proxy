"""
Tests for end-to-end encryption on the embeddings endpoint.

These tests verify that:
1. The embeddings endpoint can decrypt encrypted input
2. The embeddings response is properly encrypted
3. Both ECDSA and Ed25519 encryption are supported
"""

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
from app.api.v1.openai import VLLM_EMBEDDINGS_URL

client = TestClient(app)


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


# ==================== Embeddings Endpoint Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_ecdsa(respx_mock):
    """Test encrypted embeddings request with ECDSA."""
    # Encrypt the request input
    plain_input = "The food was delicious and the waiter was friendly."
    encrypted_input = encrypt_content(plain_input, ECDSA)

    request_data = {
        "input": encrypted_input,
        "model": "text-embedding-ada-002",
        "encoding_format": "float",
    }

    # Mock embeddings response
    response_id = "emb-encrypted-ecdsa-123"
    embedding_vector = [0.0023064255, -0.009327292, 0.015797734, 0.123456, -0.789012]
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding_vector,
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 12, "total_tokens": 12},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
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

    # Verify embedding is encrypted (should be hex string, not array)
    encrypted_embedding = response_json["data"][0]["embedding"]
    assert isinstance(encrypted_embedding, str)
    assert len(encrypted_embedding) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_embedding)

    # Verify the request was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["input"] == plain_input

    # Verify we can decrypt the embedding back to original
    decrypted_embedding_json = decrypt_content(encrypted_embedding, ECDSA)
    decrypted_embedding = json.loads(decrypted_embedding_json)
    assert decrypted_embedding == embedding_vector


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_ed25519(respx_mock):
    """Test encrypted embeddings request with Ed25519."""
    # Encrypt the request input
    plain_input = "Artificial intelligence is transforming industries."
    encrypted_input = encrypt_content(plain_input, ED25519)

    request_data = {
        "input": encrypted_input,
        "model": "text-embedding-ada-002",
    }

    # Mock embeddings response
    response_id = "emb-encrypted-ed25519-456"
    embedding_vector = [0.111, -0.222, 0.333, -0.444, 0.555]
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding_vector,
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
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

    # Verify embedding is encrypted
    encrypted_embedding = response_json["data"][0]["embedding"]
    assert isinstance(encrypted_embedding, str)
    assert len(encrypted_embedding) >= 64
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_embedding)

    # Verify the request was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["input"] == plain_input

    # Verify we can decrypt the embedding back to original
    decrypted_embedding_json = decrypt_content(encrypted_embedding, ED25519)
    decrypted_embedding = json.loads(decrypted_embedding_json)
    assert decrypted_embedding == embedding_vector


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_array_input_ecdsa(respx_mock):
    """Test encrypted embeddings with array of strings input (ECDSA)."""
    # Encrypt each input in the array
    plain_inputs = ["First text to embed", "Second text to embed"]
    encrypted_inputs = [encrypt_content(text, ECDSA) for text in plain_inputs]

    request_data = {
        "input": encrypted_inputs,
        "model": "text-embedding-ada-002",
    }

    # Mock embeddings response with multiple embeddings
    response_id = "emb-array-ecdsa-789"
    embedding_vectors = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
    ]
    response_data = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding_vectors[0],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": embedding_vectors[1],
                "index": 1,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
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

    # Verify both embeddings are encrypted
    assert len(response_json["data"]) == 2
    for i, item in enumerate(response_json["data"]):
        encrypted_embedding = item["embedding"]
        assert isinstance(encrypted_embedding, str)
        assert all(c in "0123456789abcdefABCDEF" for c in encrypted_embedding)

        # Verify we can decrypt the embedding back to original
        decrypted_embedding_json = decrypt_content(encrypted_embedding, ECDSA)
        decrypted_embedding = json.loads(decrypted_embedding_json)
        assert decrypted_embedding == embedding_vectors[i]

    # Verify the request was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["input"] == plain_inputs


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_array_input_ed25519(respx_mock):
    """Test encrypted embeddings with array of strings input (Ed25519)."""
    # Encrypt each input in the array
    plain_inputs = ["Hello world", "Goodbye world", "Testing embeddings"]
    encrypted_inputs = [encrypt_content(text, ED25519) for text in plain_inputs]

    request_data = {
        "input": encrypted_inputs,
        "model": "text-embedding-3-small",
    }

    # Mock embeddings response
    response_id = "emb-array-ed25519-abc"
    embedding_vectors = [
        [0.01, 0.02, 0.03],
        [0.04, 0.05, 0.06],
        [0.07, 0.08, 0.09],
    ]
    response_data = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding_vectors[0], "index": 0},
            {"object": "embedding", "embedding": embedding_vectors[1], "index": 1},
            {"object": "embedding", "embedding": embedding_vectors[2], "index": 2},
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
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

    # Verify all embeddings are encrypted
    assert len(response_json["data"]) == 3
    for i, item in enumerate(response_json["data"]):
        encrypted_embedding = item["embedding"]
        assert isinstance(encrypted_embedding, str)

        # Verify we can decrypt the embedding back to original
        decrypted_embedding_json = decrypt_content(encrypted_embedding, ED25519)
        decrypted_embedding = json.loads(decrypted_embedding_json)
        assert decrypted_embedding == embedding_vectors[i]

    # Verify the request was decrypted before sending to vLLM
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["input"] == plain_inputs


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_invalid_signing_algo(respx_mock):
    """Test that invalid signing algorithm is rejected."""
    request_data = {
        "input": "Some text",
        "model": "text-embedding-ada-002",
    }

    # Make request with invalid signing algorithm
    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "invalid-algo",
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    # Verify error response
    assert response.status_code == 400
    assert "Invalid X-Signing-Algo" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_missing_pub_key(respx_mock):
    """Test that missing public key means encryption is not enabled (passthrough)."""
    request_data = {
        "input": "Some text",
        "model": "text-embedding-ada-002",
    }

    # Mock embeddings response
    response_id = "emb-partial-headers-123"
    embedding_vector = [0.1, 0.2, 0.3]
    response_data = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding_vector, "index": 0}
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    from unittest.mock import patch

    with patch("app.api.v1.openai.cache"):
        # Make request with signing algo but no public key
        # When only one header is provided, encryption is not enabled
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                # Missing X-Client-Pub-Key
            },
        )

        # Verify request succeeds (encryption not enabled, passthrough mode)
        assert response.status_code == 200
        assert route.called
        response_json = response.json()

        # Verify embedding is NOT encrypted (should be array, not hex string)
        # because encryption is only enabled when BOTH headers are present
        embedding = response_json["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert embedding == embedding_vector


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_invalid_pub_key_format(respx_mock):
    """Test that invalid public key format is rejected."""
    request_data = {
        "input": "Some text",
        "model": "text-embedding-ada-002",
    }

    # Make request with invalid public key format
    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "not-a-valid-hex-string!@#$",
        },
    )

    # Verify error response
    assert response.status_code == 400
    assert "valid hex string" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_wrong_key_length_ecdsa(respx_mock):
    """Test that wrong key length for ECDSA is rejected."""
    request_data = {
        "input": "Some text",
        "model": "text-embedding-ada-002",
    }

    # Make request with wrong key length for ECDSA (should be 64 or 65 bytes)
    wrong_key = "aa" * 32  # 32 bytes instead of 64
    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": wrong_key,
        },
    )

    # Verify error response
    assert response.status_code == 400
    assert "128 hex characters" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_wrong_key_length_ed25519(respx_mock):
    """Test that wrong key length for Ed25519 is rejected."""
    request_data = {
        "input": "Some text",
        "model": "text-embedding-ada-002",
    }

    # Make request with wrong key length for Ed25519 (should be 32 bytes)
    wrong_key = "bb" * 64  # 64 bytes instead of 32
    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": wrong_key,
        },
    )

    # Verify error response
    assert response.status_code == 400
    assert "64 hex characters (32 bytes)" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_decryption_failure(respx_mock):
    """Test that invalid encrypted input causes decryption failure."""
    # Use invalid encrypted data (not properly encrypted)
    invalid_encrypted_input = "aa" * 100  # Random hex, not properly encrypted

    request_data = {
        "input": invalid_encrypted_input,
        "model": "text-embedding-ada-002",
    }

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    # Verify error response
    assert response.status_code == 400
    assert "Failed to decrypt" in response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_with_request_hash(respx_mock):
    """Test encrypted embeddings with X-Request-Hash header."""
    # Encrypt the request input
    plain_input = "Test with request hash"
    encrypted_input = encrypt_content(plain_input, ECDSA)

    request_data = {
        "input": encrypted_input,
        "model": "text-embedding-ada-002",
    }

    # Mock embeddings response
    response_id = "emb-hash-test-123"
    embedding_vector = [0.1, 0.2, 0.3]
    response_data = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding_vector, "index": 0}
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Calculate request hash
    request_hash = sha256(json.dumps(request_data).encode()).hexdigest()

    # Make request with encryption headers and request hash
    from unittest.mock import patch

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
                "X-Request-Hash": request_hash,
            },
        )

        # Verify response
        assert response.status_code == 200
        assert route.called

        # Verify that the client-provided hash was logged
        mock_log.info.assert_called_with(
            f"Using client-provided request hash: {request_hash}"
        )

        # Verify cache was called
        mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_no_encryption_headers(respx_mock):
    """Test that embeddings work normally without encryption headers."""
    plain_input = "Plain text without encryption"

    request_data = {
        "input": plain_input,
        "model": "text-embedding-ada-002",
    }

    # Mock embeddings response
    response_id = "emb-plain-123"
    embedding_vector = [0.1, 0.2, 0.3]
    response_data = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding_vector, "index": 0}
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    from unittest.mock import patch

    with patch("app.api.v1.openai.cache"):
        # Make request without encryption headers
        response = client.post(
            "/v1/embeddings",
            json=request_data,
            headers={"Authorization": TEST_AUTH_HEADER},
        )

        # Verify response
        assert response.status_code == 200
        assert route.called
        response_json = response.json()

        # Verify embedding is NOT encrypted (should be array, not hex string)
        embedding = response_json["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert embedding == embedding_vector

        # Verify the request was sent as-is
        call_args = route.calls[0].request
        sent_data = json.loads(call_args.content)
        assert sent_data["input"] == plain_input


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_embeddings_with_dimensions_param(respx_mock):
    """Test encrypted embeddings with dimensions parameter."""
    plain_input = "Test with dimensions"
    encrypted_input = encrypt_content(plain_input, ED25519)

    request_data = {
        "input": encrypted_input,
        "model": "text-embedding-3-large",
        "dimensions": 256,
    }

    # Mock embeddings response with specified dimensions
    response_id = "emb-dimensions-123"
    embedding_vector = [0.1] * 256  # 256 dimensions
    response_data = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding_vector, "index": 0}
        ],
        "model": "text-embedding-3-large",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
        "id": response_id,
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    # Make request with encryption headers
    response = client.post(
        "/v1/embeddings",
        json=request_data,
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

    # Verify embedding is encrypted
    encrypted_embedding = response_json["data"][0]["embedding"]
    assert isinstance(encrypted_embedding, str)

    # Verify we can decrypt the embedding and it has correct dimensions
    decrypted_embedding_json = decrypt_content(encrypted_embedding, ED25519)
    decrypted_embedding = json.loads(decrypted_embedding_json)
    assert len(decrypted_embedding) == 256
    assert decrypted_embedding == embedding_vector

    # Verify dimensions parameter was passed through
    call_args = route.calls[0].request
    sent_data = json.loads(call_args.content)
    assert sent_data["dimensions"] == 256
