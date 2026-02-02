"""
Tests for end-to-end encryption on the score endpoint.

These tests verify that:
1. The score endpoint can decrypt encrypted text_1 and text_2
2. The score response is properly encrypted (score field)
3. Both ECDSA and Ed25519 encryption are supported
"""

import httpx
import json
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
from app.api.v1.openai import VLLM_SCORE_URL

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


# ==================== Score Endpoint Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_ecdsa(respx_mock):
    """Test encrypted score request with ECDSA."""
    # Plain text values
    plain_text_1 = "What is the capital of France?"
    plain_text_2 = "The capital of France is Paris."

    # Encrypt texts
    encrypted_text_1 = encrypt_content(plain_text_1, ECDSA)
    encrypted_text_2 = encrypt_content(plain_text_2, ECDSA)

    # Mock response data
    response_id = "score-encrypted-ecdsa-123"
    response_data = {
        "id": response_id,
        "score": 0.95,
        "model": "test-model",
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
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

    # Verify score is encrypted (hex string)
    encrypted_score = response_json["score"]
    assert isinstance(encrypted_score, str)
    assert len(encrypted_score) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_score)

    # Verify the texts were decrypted before sending to vLLM
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["text_1"] == plain_text_1
    assert sent_body["text_2"] == plain_text_2

    # Verify we can decrypt the response score back to original
    decrypted_score = decrypt_content(encrypted_score, ECDSA)
    assert json.loads(decrypted_score) == 0.95


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_ed25519(respx_mock):
    """Test encrypted score request with Ed25519."""
    # Plain text values
    plain_text_1 = "How do neural networks work?"
    plain_text_2 = "Neural networks are computational models inspired by the brain."

    # Encrypt texts
    encrypted_text_1 = encrypt_content(plain_text_1, ED25519)
    encrypted_text_2 = encrypt_content(plain_text_2, ED25519)

    # Mock response data
    response_id = "score-encrypted-ed25519-456"
    response_data = {
        "id": response_id,
        "score": 0.88,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
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

    # Verify score is encrypted
    encrypted_score = response_json["score"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_score)

    # Verify we can decrypt it
    decrypted_score = decrypt_content(encrypted_score, ED25519)
    assert json.loads(decrypted_score) == 0.88


@pytest.mark.asyncio
@pytest.mark.respx
async def test_score_no_encryption(respx_mock):
    """Test score request without encryption headers."""
    plain_text_1 = "Test query"
    plain_text_2 = "Test document"

    response_id = "score-no-enc"
    response_data = {
        "id": response_id,
        "score": 0.7,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": plain_text_1,
                "text_2": plain_text_2,
            },
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    response_json = response.json()

    # Without encryption, score should be plain number
    assert response_json["score"] == 0.7


@pytest.mark.asyncio
async def test_encrypted_score_invalid_signing_algo():
    """Test score request with invalid signing algorithm."""
    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": "query",
            "text_2": "doc",
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": "invalid-algo",
            "X-Client-Pub-Key": "0" * 64,
        },
    )

    assert response.status_code == 400
    assert "Invalid X-Signing-Algo" in response.json()["detail"]


@pytest.mark.asyncio
async def test_encrypted_score_invalid_pub_key_format():
    """Test score request with invalid public key format."""
    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": "query",
            "text_2": "doc",
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "not-valid-hex!@#$",
        },
    )

    assert response.status_code == 400
    assert "valid hex string" in response.json()["detail"]


@pytest.mark.asyncio
async def test_encrypted_score_wrong_key_length_ecdsa():
    """Test score request with wrong ECDSA public key length."""
    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": "query",
            "text_2": "doc",
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": "0" * 32,  # Too short
        },
    )

    assert response.status_code == 400
    assert "ECDSA public key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_encrypted_score_wrong_key_length_ed25519():
    """Test score request with wrong Ed25519 public key length."""
    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": "query",
            "text_2": "doc",
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ED25519,
            "X-Client-Pub-Key": "0" * 32,  # Too short (should be 64)
        },
    )

    assert response.status_code == 400
    assert "Ed25519 public key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_encrypted_score_decryption_failure_text_1():
    """Test score request with text_1 that cannot be decrypted."""
    # Invalid encrypted data (valid hex but won't decrypt)
    invalid_encrypted = "0" * 128

    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": invalid_encrypted,
            "text_2": encrypt_content("valid doc", ECDSA),
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400
    assert "decrypt" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_encrypted_score_decryption_failure_text_2():
    """Test score request with text_2 that cannot be decrypted."""
    # Invalid encrypted data (valid hex but won't decrypt)
    invalid_encrypted = "0" * 128

    response = client.post(
        "/v1/score",
        json={
            "model": "test-model",
            "text_1": encrypt_content("valid query", ECDSA),
            "text_2": invalid_encrypted,
        },
        headers={
            "Authorization": TEST_AUTH_HEADER,
            "X-Signing-Algo": ECDSA,
            "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
        },
    )

    assert response.status_code == 400
    assert "decrypt" in response.json()["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_with_request_hash(respx_mock):
    """Test encrypted score request with X-Request-Hash header."""
    encrypted_text_1 = encrypt_content("test query", ECDSA)
    encrypted_text_2 = encrypt_content("test doc", ECDSA)

    expected_hash = "custom-encrypted-score-hash"
    response_id = "score-hash"
    response_data = {
        "id": response_id,
        "score": 0.9,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
                "X-Request-Hash": expected_hash,
            },
        )

    assert response.status_code == 200
    mock_log.info.assert_called_with(
        f"Using client-provided request hash: {expected_hash}"
    )
    mock_cache.set_chat.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_partial_headers(respx_mock):
    """Test score request with only one encryption header (should work without encryption)."""
    response_data = {
        "id": "score-partial",
        "score": 0.8,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        # Only X-Signing-Algo, no X-Client-Pub-Key
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": "plain query",
                "text_2": "plain doc",
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                # Missing X-Client-Pub-Key
            },
        )

    # Should proceed without encryption (partial headers = no encryption)
    # Score should be plain number, not encrypted
    assert response.status_code == 200
    assert route.called
    result = response.json()
    assert result["score"] == 0.8  # Plain number, not encrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_generates_id_if_missing(respx_mock):
    """Test that encrypted score generates ID if not in response."""
    encrypted_text_1 = encrypt_content("test query", ECDSA)
    encrypted_text_2 = encrypt_content("test doc", ECDSA)

    # Response without ID
    response_data = {
        "score": 0.75,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["id"].startswith("score-")
    assert len(result["id"]) == 30  # "score-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_with_negative_score(respx_mock):
    """Test encrypted score with negative score value."""
    encrypted_text_1 = encrypt_content("unrelated query", ECDSA)
    encrypted_text_2 = encrypt_content("completely different", ECDSA)

    response_id = "score-negative-enc"
    response_data = {
        "id": response_id,
        "score": -0.5,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    response_json = response.json()

    # Verify we can decrypt the negative score
    decrypted_score = decrypt_content(response_json["score"], ECDSA)
    assert json.loads(decrypted_score) == -0.5


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_score_with_null_score(respx_mock):
    """Test encrypted score when score is null (should not encrypt null)."""
    encrypted_text_1 = encrypt_content("query", ECDSA)
    encrypted_text_2 = encrypt_content("doc", ECDSA)

    response_id = "score-null"
    response_data = {
        "id": response_id,
        "score": None,
        "model": "test-model",
    }

    route = respx_mock.post(VLLM_SCORE_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/score",
            json={
                "model": "test-model",
                "text_1": encrypted_text_1,
                "text_2": encrypted_text_2,
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    response_json = response.json()
    # Null score should remain null (not encrypted)
    assert response_json["score"] is None
