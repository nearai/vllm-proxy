"""
Encryption utilities for end-to-end encryption using ECDSA and Ed25519 signing keys.

For ECDSA: Uses ECIES (Elliptic Curve Integrated Encryption Scheme)
For Ed25519: Uses X25519 key exchange + AES-GCM encryption
"""

import json
import os
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from app.logger import log
from app.quote.quote import (
    SigningContext,
    ECDSA,
    ED25519,
    ecdsa_context,
    ed25519_context,
)


def _ed25519_to_x25519_private_key(
    ed25519_private: ed25519.Ed25519PrivateKey,
) -> X25519PrivateKey:
    """
    Convert Ed25519 private key to X25519 private key.
    
    Both Ed25519 and X25519 use Curve25519, but with different representations.
    The private key (seed) can be shared, but needs proper clamping for X25519.
    """
    # Get the raw private key bytes (seed)
    private_bytes = ed25519_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    # X25519 requires clamping: clear bits 0, 1, 2, and set bit 254
    # This ensures the scalar is a valid X25519 private key
    clamped = bytearray(private_bytes)
    clamped[0] &= 0xF8  # Clear bottom 3 bits
    clamped[31] &= 0x7F  # Clear top bit
    clamped[31] |= 0x40  # Set second-highest bit
    
    return X25519PrivateKey.from_private_bytes(bytes(clamped))


def _ed25519_to_x25519_public_key(ed25519_public_bytes: bytes) -> X25519PublicKey:
    """
    Convert Ed25519 public key (Edwards coordinates) to X25519 public key (Montgomery coordinates).
    
    Ed25519 uses Edwards form: (x, y) on curve edwards25519
    X25519 uses Montgomery form: u on curve montgomery25519
    
    Both curves are birationally equivalent forms of Curve25519.
    
    Conversion formula: u = (1 + y) / (1 - y) mod p
    where p = 2^255 - 19
    
    Ed25519 public key format: 32 bytes encoding y-coordinate (little-endian)
    with sign bit of x in the most significant bit of the last byte.
    """
    # Ed25519 public key bytes encode the y-coordinate in compressed format
    # The format is: y (255 bits, little-endian) with sign(x) in bit 255
    
    # Extract y-coordinate (clear the sign bit)
    y_bytes = bytearray(ed25519_public_bytes)
    y_bytes[31] &= 0x7F  # Clear the sign bit (most significant bit)
    
    # Convert y-coordinate from little-endian bytes to integer
    y = int.from_bytes(y_bytes, byteorder="little")
    
    # Curve25519 prime: p = 2^255 - 19
    p = 2**255 - 19
    
    # Convert from Edwards to Montgomery: u = (1 + y) / (1 - y) mod p
    # Handle edge case: if y == 1, then 1 - y == 0, which would cause division by zero
    # In practice, y == 1 corresponds to the identity point, which maps to u = 0
    if y == 1:
        u = 0
    else:
        # Compute 1 - y mod p
        one_minus_y = (1 - y) % p
        # Compute modular inverse using Fermat's little theorem: a^(p-2) mod p
        inv_one_minus_y = pow(one_minus_y, p - 2, p)
        # Compute u = (1 + y) * inv(1 - y) mod p
        u = ((1 + y) * inv_one_minus_y) % p
    
    # Convert u to bytes (little-endian, 32 bytes)
    u_bytes = u.to_bytes(32, byteorder="little")
    
    return X25519PublicKey.from_public_bytes(u_bytes)


def encrypt_data(data: bytes, public_key_hex: str, signing_algo: str) -> bytes:
    """
    Encrypt data using the provided public key.

    Args:
        data: Data to encrypt
        public_key_hex: Public key in hex format
        signing_algo: Either 'ecdsa' or 'ed25519'

    Returns:
        Encrypted data as bytes (format: [nonce (12 bytes)][encrypted_data])
    """
    if signing_algo == ED25519:
        return _encrypt_ed25519(data, public_key_hex)
    elif signing_algo == ECDSA:
        return _encrypt_ecdsa(data, public_key_hex)
    else:
        raise ValueError(f"Unsupported signing algorithm: {signing_algo}")


def decrypt_data(encrypted_data: bytes, context: SigningContext) -> bytes:
    """
    Decrypt data using the signing context's private key.

    Args:
        encrypted_data: Encrypted data (format: [nonce (12 bytes)][encrypted_data])
        context: SigningContext with private key

    Returns:
        Decrypted data as bytes
    """
    if context.method == ED25519:
        return _decrypt_ed25519(encrypted_data, context)
    elif context.method == ECDSA:
        return _decrypt_ecdsa(encrypted_data, context)
    else:
        raise ValueError(f"Unsupported signing algorithm: {context.method}")


def _encrypt_ed25519(data: bytes, public_key_hex: str) -> bytes:
    """Encrypt data using Ed25519 public key (via X25519 key exchange)."""
    try:
        # Parse public key from hex
        public_key_bytes = bytes.fromhex(public_key_hex)
        if len(public_key_bytes) != 32:
            raise ValueError(
                f"Ed25519 public key must be 32 bytes, got {len(public_key_bytes)}"
            )

        # Convert to X25519 public key
        x25519_public = _ed25519_to_x25519_public_key(public_key_bytes)

        # Generate ephemeral X25519 key pair
        ephemeral_private = X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()

        # Perform key exchange
        shared_secret = ephemeral_private.exchange(x25519_public)

        # Derive AES key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"ed25519_encryption",
            backend=default_backend(),
        )
        aes_key = hkdf.derive(shared_secret)

        # Encrypt with AES-GCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, data, None)

        # Format: [ephemeral_public_key (32 bytes)][nonce (12 bytes)][ciphertext]
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return ephemeral_public_bytes + nonce + ciphertext
    except Exception as e:
        log.error(f"Ed25519 encryption failed: {e}")
        raise


def _decrypt_ed25519(encrypted_data: bytes, context: SigningContext) -> bytes:
    """Decrypt data using Ed25519 private key."""
    try:
        if context._ed_private is None:
            raise ValueError("Ed25519 context not properly initialized")

        if (
            len(encrypted_data) < 44
        ):  # 32 (ephemeral public) + 12 (nonce) + at least some ciphertext
            raise ValueError("Encrypted data too short")

        # Extract components
        ephemeral_public_bytes = encrypted_data[:32]
        nonce = encrypted_data[32:44]
        ciphertext = encrypted_data[44:]

        # Convert Ed25519 private to X25519 private
        x25519_private = _ed25519_to_x25519_private_key(context._ed_private)

        # Convert ephemeral public key to X25519
        ephemeral_public = X25519PublicKey.from_public_bytes(ephemeral_public_bytes)

        # Perform key exchange
        shared_secret = x25519_private.exchange(ephemeral_public)

        # Derive AES key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"ed25519_encryption",
            backend=default_backend(),
        )
        aes_key = hkdf.derive(shared_secret)

        # Decrypt with AES-GCM
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return plaintext
    except Exception as e:
        log.error(f"Ed25519 decryption failed: {e}")
        raise


def _encrypt_ecdsa(data: bytes, public_key_hex: str) -> bytes:
    """Encrypt data using ECDSA public key (ECIES)."""
    try:
        # Parse public key from hex (uncompressed format: 0x04 + 64 bytes)
        public_key_bytes = bytes.fromhex(public_key_hex)
        if len(public_key_bytes) == 65 and public_key_bytes[0] == 0x04:
            # Uncompressed format
            public_key_bytes = public_key_bytes[1:]  # Remove 0x04 prefix

        if len(public_key_bytes) != 64:
            raise ValueError(
                f"ECDSA public key must be 64 bytes (or 65 with 0x04), got {len(public_key_bytes)}"
            )

        # Reconstruct public key
        x = int.from_bytes(public_key_bytes[:32], "big")
        y = int.from_bytes(public_key_bytes[32:], "big")

        # Create EC public key
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), b"\x04" + public_key_bytes
        )

        # Generate ephemeral EC key pair
        ephemeral_private = ec.generate_private_key(ec.SECP256K1(), default_backend())
        ephemeral_public = ephemeral_private.public_key()

        # Perform ECDH key exchange
        shared_secret = ephemeral_private.exchange(ec.ECDH(), public_key)

        # Derive AES key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"ecdsa_encryption",
            backend=default_backend(),
        )
        aes_key = hkdf.derive(shared_secret)

        # Encrypt with AES-GCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, data, None)

        # Format: [ephemeral_public_key (65 bytes uncompressed)][nonce (12 bytes)][ciphertext]
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        return ephemeral_public_bytes + nonce + ciphertext
    except Exception as e:
        log.error(f"ECDSA encryption failed: {e}")
        raise


def _decrypt_ecdsa(encrypted_data: bytes, context: SigningContext) -> bytes:
    """Decrypt data using ECDSA private key."""
    try:
        if context._raw_account is None:
            raise ValueError("ECDSA context not properly initialized")

        if (
            len(encrypted_data) < 77
        ):  # 65 (ephemeral public) + 12 (nonce) + at least some ciphertext
            raise ValueError("Encrypted data too short")

        # Extract components
        ephemeral_public_bytes = encrypted_data[:65]
        nonce = encrypted_data[65:77]
        ciphertext = encrypted_data[77:]

        # Parse ephemeral public key
        ephemeral_public = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), ephemeral_public_bytes
        )

        # Get private key bytes from account
        # web3 Account.key is an eth_keys.PrivateKey object
        private_key_bytes = context._raw_account.key.to_bytes()

        # Create EC private key from bytes
        private_key = ec.derive_private_key(
            int.from_bytes(private_key_bytes, "big"), ec.SECP256K1(), default_backend()
        )

        # Perform ECDH key exchange
        shared_secret = private_key.exchange(ec.ECDH(), ephemeral_public)

        # Derive AES key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"ecdsa_encryption",
            backend=default_backend(),
        )
        aes_key = hkdf.derive(shared_secret)

        # Decrypt with AES-GCM
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return plaintext
    except Exception as e:
        log.error(f"ECDSA decryption failed: {e}")
        raise
