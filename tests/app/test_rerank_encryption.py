"""
Tests for end-to-end encryption on the rerank endpoint.

These tests verify that:
1. The rerank endpoint can decrypt encrypted query and documents
2. The rerank response document text is properly encrypted
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
from app.api.v1.openai import VLLM_RERANK_URL

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


# ==================== Rerank Endpoint Encryption Tests ====================


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_rerank_ecdsa(respx_mock):
    """Test encrypted rerank request with ECDSA."""
    # Plain text values
    plain_query = "What is the capital of France?"
    plain_docs = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ]

    # Encrypt query and documents
    encrypted_query = encrypt_content(plain_query, ECDSA)
    encrypted_docs = [encrypt_content(doc, ECDSA) for doc in plain_docs]

    # Mock response data
    response_id = "rerank-encrypted-ecdsa-123"
    response_data = {
        "id": response_id,
        "results": [
            {
                "index": 0,
                "relevance_score": 0.98,
                "document": {"text": "Paris is the capital of France."},
            },
            {
                "index": 1,
                "relevance_score": 0.05,
                "document": {"text": "Berlin is the capital of Germany."},
            },
        ],
        "model": "rerank-model",
    }

    # Setup RESPX mock
    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": encrypted_docs,
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
    assert len(response_json["results"]) == 2

    # Verify document text is encrypted (hex string)
    encrypted_text = response_json["results"][0]["document"]["text"]
    assert isinstance(encrypted_text, str)
    assert len(encrypted_text) >= 64  # Encrypted data should be long hex string
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify the query was decrypted before sending to vLLM
    call_args = route.calls[0].request
    import json
    sent_body = json.loads(call_args.content)
    assert sent_body["query"] == plain_query
    assert sent_body["documents"] == plain_docs

    # Verify we can decrypt the response document text back to original
    decrypted_text = decrypt_content(encrypted_text, ECDSA)
    assert decrypted_text == "Paris is the capital of France."


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_rerank_ed25519(respx_mock):
    """Test encrypted rerank request with Ed25519."""
    # Plain text values
    plain_query = "What programming languages are popular?"
    plain_docs = [
        "Python is widely used for data science.",
        "JavaScript is the language of the web.",
    ]

    # Encrypt query and documents
    encrypted_query = encrypt_content(plain_query, ED25519)
    encrypted_docs = [encrypt_content(doc, ED25519) for doc in plain_docs]

    # Mock response data
    response_id = "rerank-encrypted-ed25519-456"
    response_data = {
        "id": response_id,
        "results": [
            {
                "index": 0,
                "relevance_score": 0.90,
                "document": {"text": "Python is widely used for data science."},
            },
            {
                "index": 1,
                "relevance_score": 0.85,
                "document": {"text": "JavaScript is the language of the web."},
            },
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": encrypted_docs,
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

    # Verify document text is encrypted
    encrypted_text = response_json["results"][0]["document"]["text"]
    assert all(c in "0123456789abcdefABCDEF" for c in encrypted_text)

    # Verify we can decrypt it
    decrypted_text = decrypt_content(encrypted_text, ED25519)
    assert decrypted_text == "Python is widely used for data science."


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_rerank_with_document_objects(respx_mock):
    """Test encrypted rerank request with document objects (text field)."""
    plain_query = "Find relevant documents"
    plain_doc_text = "This is a relevant document."

    encrypted_query = encrypt_content(plain_query, ECDSA)
    encrypted_doc_text = encrypt_content(plain_doc_text, ECDSA)

    response_id = "rerank-doc-obj"
    response_data = {
        "id": response_id,
        "results": [
            {
                "index": 0,
                "relevance_score": 0.95,
                "document": {"text": "This is a relevant document."},
            },
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": [{"text": encrypted_doc_text}],  # Document as object
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    assert route.called

    # Verify the document text was decrypted before sending to vLLM
    import json
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["documents"][0]["text"] == plain_doc_text


@pytest.mark.asyncio
@pytest.mark.respx
async def test_rerank_no_encryption(respx_mock):
    """Test rerank request without encryption headers."""
    plain_query = "Test query"
    plain_docs = ["Document 1", "Document 2"]

    response_id = "rerank-no-enc"
    response_data = {
        "id": response_id,
        "results": [
            {"index": 0, "relevance_score": 0.9, "document": {"text": "Document 1"}},
            {"index": 1, "relevance_score": 0.5, "document": {"text": "Document 2"}},
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": plain_query,
                "documents": plain_docs,
            },
            headers={"Authorization": TEST_AUTH_HEADER},
        )

    assert response.status_code == 200
    response_json = response.json()

    # Without encryption, document text should be plain
    assert response_json["results"][0]["document"]["text"] == "Document 1"
    assert response_json["results"][1]["document"]["text"] == "Document 2"


@pytest.mark.asyncio
async def test_encrypted_rerank_invalid_signing_algo():
    """Test rerank request with invalid signing algorithm."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-model",
            "query": "test",
            "documents": ["doc1"],
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
async def test_encrypted_rerank_invalid_pub_key_format():
    """Test rerank request with invalid public key format."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-model",
            "query": "test",
            "documents": ["doc1"],
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
async def test_encrypted_rerank_wrong_key_length_ecdsa():
    """Test rerank request with wrong ECDSA public key length."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-model",
            "query": "test",
            "documents": ["doc1"],
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
async def test_encrypted_rerank_wrong_key_length_ed25519():
    """Test rerank request with wrong Ed25519 public key length."""
    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-model",
            "query": "test",
            "documents": ["doc1"],
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
async def test_encrypted_rerank_decryption_failure():
    """Test rerank request with data that cannot be decrypted."""
    # Invalid encrypted data (valid hex but won't decrypt)
    invalid_encrypted = "0" * 128

    response = client.post(
        "/v1/rerank",
        json={
            "model": "rerank-model",
            "query": invalid_encrypted,
            "documents": ["doc1"],
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
async def test_encrypted_rerank_with_request_hash(respx_mock):
    """Test encrypted rerank request with X-Request-Hash header."""
    plain_query = "Test query"
    encrypted_query = encrypt_content(plain_query, ECDSA)

    expected_hash = "custom-encrypted-rerank-hash"
    response_id = "rerank-hash"
    response_data = {
        "id": response_id,
        "results": [{"index": 0, "relevance_score": 0.9}],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache") as mock_cache, patch(
        "app.api.v1.openai.log"
    ) as mock_log:
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": [encrypt_content("doc", ECDSA)],
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
async def test_encrypted_rerank_partial_headers(respx_mock):
    """Test rerank request with only one encryption header (should work without encryption)."""
    response_data = {
        "id": "rerank-partial",
        "results": [{"index": 0, "relevance_score": 0.9, "document": {"text": "doc1"}}],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        # Only X-Signing-Algo, no X-Client-Pub-Key
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": "plain query",
                "documents": ["doc1"],
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                # Missing X-Client-Pub-Key
            },
        )

    # Should proceed without encryption (partial headers = no encryption)
    # Response document text should be plain (not encrypted)
    assert response.status_code == 200
    assert route.called
    result = response.json()
    assert result["results"][0]["document"]["text"] == "doc1"  # Plain text, not encrypted


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_rerank_generates_id_if_missing(respx_mock):
    """Test that encrypted rerank generates ID if not in response."""
    encrypted_query = encrypt_content("test query", ECDSA)

    # Response without ID
    response_data = {
        "results": [
            {"index": 0, "relevance_score": 0.9, "document": {"text": "doc text"}}
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": [encrypt_content("doc", ECDSA)],
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["id"].startswith("rerank-")
    assert len(result["id"]) == 31  # "rerank-" + 24 hex chars


@pytest.mark.asyncio
@pytest.mark.respx
async def test_encrypted_rerank_without_return_documents(respx_mock):
    """Test encrypted rerank when return_documents=false (no document text to encrypt)."""
    encrypted_query = encrypt_content("test query", ECDSA)

    response_id = "rerank-no-docs"
    response_data = {
        "id": response_id,
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5},
        ],
        "model": "rerank-model",
    }

    route = respx_mock.post(VLLM_RERANK_URL).mock(
        return_value=httpx.Response(200, json=response_data)
    )

    with patch("app.api.v1.openai.cache"):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "rerank-model",
                "query": encrypted_query,
                "documents": [encrypt_content("doc", ECDSA)],
                "return_documents": False,
            },
            headers={
                "Authorization": TEST_AUTH_HEADER,
                "X-Signing-Algo": ECDSA,
                "X-Client-Pub-Key": real_ecdsa_context.signing_public_key,
            },
        )

    assert response.status_code == 200
    result = response.json()
    # No document field in results
    assert "document" not in result["results"][0]
    assert "document" not in result["results"][1]
