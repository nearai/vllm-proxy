# Confidential AI Verifier

Tools for validating Phala Cloud attestation and response signatures.

## Requirements

- Python 3.10+
- `requests`, `eth-account`, `cryptography`, `pynacl`
- Phala Cloud API key from https://redpill.ai (for signature verifier and encryption verifier)

## Attestation Verifier

Generates a fresh nonce, requests a new attestation, and verifies:
- **GPU attestation**: Submits GPU evidence payload to NVIDIA NRAS (https://nras.attestation.nvidia.com) and verifies the nonce matches
- **TDX report data**: Validates that report data binds the signing key (ECDSA or Ed25519) and nonce
- **Intel TDX quote**: Verifies TDX quote via Phala's verification service (https://cloud-api.phala.network)
- **Compose manifest**: Displays Docker compose manifest and verifies it matches the mr_config measurement

### Usage

```bash
cd verifiers
python3 attestation_verifier.py [--model MODEL_NAME]
```

Default model: `phala/deepseek-chat-v3-0324`

No API key required. The verifier fetches attestations from the public `/v1/attestation/report` endpoint.

### Example Output

```
Signing address: 0x1234...
Request nonce: abc123...

🔐 TDX report data
Signing algorithm: ecdsa
Report data binds signing address: True
Report data embeds request nonce: True

🔐 GPU attestation
GPU payload nonce matches request_nonce: True
NVIDIA attestation verdict: PASS

🔐 Intel TDX quote
Intel TDX quote verified: True
```

## Signature Verifier

Fetches chat completions (streaming and non-streaming), verifies ECDSA signatures, and validates attestations:
1. Sends chat completion request to `/v1/chat/completions`
2. Fetches signature from `/v1/signature/{chat_id}` endpoint
3. Verifies request hash and response hash match the signed hashes
4. Recovers ECDSA signing address from signature
5. Fetches fresh attestation with user-supplied nonce for the recovered signing address
6. Validates attestation using the same checks as attestation verifier

**Note**: The verifier supplies a fresh nonce when fetching attestation (step 5), which ensures attestation freshness but means the nonce/report_data won't match the original signing context. This is expected behavior - the verifier proves the signing key is bound to valid hardware, not that a specific attestation was used for signing.

### Setup

Set your API key as an environment variable:

```bash
export API_KEY=your-api-key-here
```

Or create a `.env` file:

```bash
API_KEY=your-api-key-here
```

Then run from the verifiers directory:

```bash
cd verifiers
source .env
python3 signature_verifier.py [--model MODEL_NAME]
```

Default model: `phala/deepseek-chat-v3-0324`

### What It Verifies

- Request body hash matches server-computed hash
- Response text hash matches server-computed hash
- ECDSA signature is valid and recovers to the claimed signing address
- Signing address is bound to hardware via TDX report data
- GPU attestation passes NVIDIA verification
- Intel TDX quote is valid

## Sigstore Provenance

Both scripts automatically extract all container image digests from the Docker compose manifest (matching `@sha256:xxx` patterns) and verify Sigstore accessibility for each image. This allows you to:

1. Verify the container images were built from the expected source repository
2. Review the GitHub Actions workflow that built the images
3. Audit the build provenance and supply chain metadata

The verifiers check each Sigstore link with an HTTP HEAD request to ensure provenance data is available (not 404).

Example output:
```
🔐 Sigstore provenance
Checking Sigstore accessibility for container images...
  ✓ https://search.sigstore.dev/?hash=sha256:77fbe5f142419d6f52b04c0e749aa3facf9359dcd843f68d073e24d0eba7c5dd (HTTP 200)
  ✓ https://search.sigstore.dev/?hash=sha256:abc123... (HTTP 200)
```

If a link returns ✗, the provenance data may not be available in Sigstore (either the image wasn't signed or the digest is incorrect).

## Multi-Server Load Balancer Setup

In production deployments with multiple backend servers behind a load balancer:

### Server Behavior
- Each server has its own unique signing key/address
- Attestation requests with `signing_address` parameter return 404 if the address doesn't match
- Response includes `all_attestations: [attestation]` (single-element array with this server's attestation)

### Load Balancer Requirements
When `/v1/attestation/report?signing_address={addr}&nonce={nonce}`:
1. **Broadcast** the request to all backend servers
2. Collect non-404 responses from servers matching the signing_address
3. Merge `all_attestations` arrays from all responses
4. Return combined response with all servers' attestations

### Verifier Flow
1. Get signature → extract `signing_address`
2. Request attestation with `signing_address` parameter
3. LB broadcasts → collect attestations from all servers
4. Verifier finds matching attestation by comparing `signing_address` in `all_attestations`

### Example Response (Multi-Server)
```json
{
  "signing_address": "0xServer1...",
  "intel_quote": "...",
  "all_attestations": [
    {"signing_address": "0xServer1...", "intel_quote": "...", ...},
    {"signing_address": "0xServer2...", "intel_quote": "...", ...}
  ]
}
```

The verifier filters `all_attestations` to find the entry matching the signature's `signing_address`.

## Encryption Verifier

Tests end-to-end encryption with vllm-proxy directly. This verifier:

1. Fetches the model's public key from `/v1/attestation/report` endpoint
2. Generates a client key pair (ECDSA or Ed25519)
3. Encrypts request message content using the model's public key
4. Sends encrypted request with encryption headers (`X-Signing-Algo`, `X-Client-Pub-Key`)
5. Receives encrypted response and decrypts it using the client's private key
6. Tests both streaming and non-streaming chat completions

### Setup

Set your API key and base URL as environment variables:

```bash
export API_KEY=your-api-key-here
export BASE_URL=http://localhost:8000  # or your vllm-proxy URL
```

Or create a `.env` file:

```bash
API_KEY=your-api-key-here
BASE_URL=http://localhost:8000
```

### Usage

```bash
cd verifiers
source .env  # if using .env file
python3 encryption_verifier.py [OPTIONS]
```

### Options

- `--model MODEL_NAME`: Model name (default: `phala/deepseek-chat-v3-0324`)
- `--base-url URL`: Base URL for vllm-proxy (overrides `BASE_URL` env var)
- `--signing-algo {ecdsa,ed25519}`: Signing algorithm to use (default: `ecdsa`)
- `--test-both`: Test both ECDSA and Ed25519 algorithms

### Examples

Test with ECDSA (default):
```bash
python3 encryption_verifier.py --model phala/deepseek-chat-v3-0324
```

Test with Ed25519:
```bash
python3 encryption_verifier.py --signing-algo ed25519
```

Test both algorithms:
```bash
python3 encryption_verifier.py --test-both
```

Test against a specific vllm-proxy instance:
```bash
python3 encryption_verifier.py --base-url http://your-vllm-proxy:8000
```

### Example Output

```
Testing against: http://localhost:8000
API Key: Set

============================================================
Encrypted Streaming Example (ECDSA)
============================================================
✓ Fetched model public key: 0123456789abcdef0123456789abcdef...
✓ Generated client key pair: fedcba9876543210fedcba9876543210...
✓ Encrypted message content
✓ Request sent successfully (HTTP 200)

Receiving stream...
✓ Chat ID: chatcmpl-abc123...
  Decrypted chunk: Hello
  Decrypted chunk: !
  Decrypted chunk:  How
  Decrypted chunk:  can
  Decrypted chunk:  I
  Decrypted chunk:  help
  Decrypted chunk:  you
  Decrypted chunk: ?

✓ Complete decrypted response: Hello! How can I help you?
✓ Total response length: 1234 bytes

============================================================
Encrypted Non-Streaming Example (ECDSA)
============================================================
✓ Fetched model public key: 0123456789abcdef0123456789abcdef...
✓ Generated client key pair: fedcba9876543210fedcba9876543210...
✓ Encrypted message content
✓ Request sent successfully (HTTP 200)
✓ Chat ID: chatcmpl-xyz789...
✓ Decrypted response: Hello! How can I help you?
```

### Encryption Details

- **ECDSA**: Uses ECIES (Elliptic Curve Integrated Encryption Scheme) with AES-GCM
- **Ed25519**: Uses X25519 key exchange + ChaCha20-Poly1305 encryption (via PyNaCl Box)

The encryption follows the same scheme as implemented in vllm-proxy:
- Request message content is encrypted as hex strings
- Response message content is encrypted as hex strings
- Only message content is encrypted, not the entire request/response body
