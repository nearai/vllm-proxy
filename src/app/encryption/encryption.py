"""
Encryption utilities for end-to-end encryption using ECDSA and Ed25519 signing keys.

For ECDSA: Uses ECIES (Elliptic Curve Integrated Encryption Scheme)
For Ed25519: Uses PyNaCl Box (X25519 key exchange + ChaCha20-Poly1305 encryption)
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
from nacl.public import PrivateKey as X25519PrivateKeyNaCl, PublicKey as X25519PublicKeyNaCl, Box
from nacl import bindings
from app.logger import log
from app.quote.quote import (
    SigningContext,
    ECDSA,
    ED25519,
    ecdsa_context,
    ed25519_context,
)


def _ed25519_to_x25519_private_key_nacl(
    ed25519_private: ed25519.Ed25519PrivateKey,
) -> X25519PrivateKeyNaCl:
    """
    Convert Ed25519 private key to PyNaCl X25519 private key using PyNaCl bindings.
    
    Uses PyNaCl's built-in conversion function which handles the clamping automatically.
    PyNaCl expects Ed25519 secret key in 64-byte format: [32-byte seed][32-byte public key]
    """
    # Get the raw private key bytes (seed) - 32 bytes
    seed_bytes = ed25519_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    # Get the public key bytes - 32 bytes
    public_key_bytes = ed25519_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    
    # PyNaCl expects Ed25519 secret key as 64 bytes: seed + public_key
    ed25519_secret_key = seed_bytes + public_key_bytes
    
    # Use PyNaCl's built-in conversion function
    # crypto_sign_ed25519_sk_to_curve25519 converts Ed25519 secret key to X25519 secret key
    x25519_private_bytes = bindings.crypto_sign_ed25519_sk_to_curve25519(ed25519_secret_key)
    
    return X25519PrivateKeyNaCl(x25519_private_bytes)


def _ed25519_to_x25519_public_key_nacl(ed25519_public_bytes: bytes) -> X25519PublicKeyNaCl:
    """
    Convert Ed25519 public key to PyNaCl X25519 public key using PyNaCl bindings.
    
    Uses PyNaCl's built-in conversion function for Edwards to Montgomery coordinate conversion.
    """
    # Use PyNaCl's built-in conversion function
    # crypto_sign_ed25519_pk_to_curve25519 converts Ed25519 public key to X25519 public key
    x25519_public_bytes = bindings.crypto_sign_ed25519_pk_to_curve25519(ed25519_public_bytes)
    
    return X25519PublicKeyNaCl(x25519_public_bytes)


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
    """Encrypt data using Ed25519 public key via PyNaCl Box (X25519 + ChaCha20-Poly1305)."""
    try:
        # Parse public key from hex
        public_key_bytes = bytes.fromhex(public_key_hex)
        if len(public_key_bytes) != 32:
            raise ValueError(
                f"Ed25519 public key must be 32 bytes, got {len(public_key_bytes)}"
            )

        # Convert Ed25519 public key to X25519 public key (PyNaCl format)
        x25519_public = _ed25519_to_x25519_public_key_nacl(public_key_bytes)

        # Generate ephemeral X25519 key pair using PyNaCl
        ephemeral_private = X25519PrivateKeyNaCl.generate()
        ephemeral_public = ephemeral_private.public_key

        # Create Box for encryption (sender uses ephemeral private, recipient uses their public)
        box = Box(ephemeral_private, x25519_public)

        # Encrypt using PyNaCl Box (automatically generates nonce and uses ChaCha20-Poly1305)
        # Box.encrypt returns: [nonce (24 bytes)][ciphertext]
        encrypted = box.encrypt(data)

        # Format: [ephemeral_public_key (32 bytes)][nonce (24 bytes)][ciphertext]
        # PyNaCl Box uses 24-byte nonce (included in encrypted output)
        ephemeral_public_bytes = bytes(ephemeral_public)
        return ephemeral_public_bytes + encrypted
    except Exception as e:
        log.error(f"Ed25519 encryption failed: {e}")
        raise


def _decrypt_ed25519(encrypted_data: bytes, context: SigningContext) -> bytes:
    """Decrypt data using Ed25519 private key via PyNaCl Box."""
    try:
        if context._ed_private is None:
            raise ValueError("Ed25519 context not properly initialized")

        # Format: [ephemeral_public_key (32 bytes)][nonce (24 bytes)][ciphertext]
        # Minimum: 32 (ephemeral public) + 24 (nonce) + 0 (empty ciphertext) = 56 bytes
        if len(encrypted_data) < 56:
            raise ValueError("Encrypted data too short")

        # Extract components
        ephemeral_public_bytes = encrypted_data[:32]
        box_encrypted = encrypted_data[32:]  # Contains [nonce (24 bytes)][ciphertext]

        # Convert Ed25519 private to X25519 private (PyNaCl format)
        x25519_private = _ed25519_to_x25519_private_key_nacl(context._ed_private)

        # Convert ephemeral public key to X25519 (PyNaCl format)
        ephemeral_public = X25519PublicKeyNaCl(ephemeral_public_bytes)

        # Create Box for decryption (recipient uses their private, sender's ephemeral public)
        box = Box(x25519_private, ephemeral_public)

        # Decrypt using PyNaCl Box (automatically extracts nonce and verifies Poly1305 tag)
        plaintext = box.decrypt(box_encrypted)

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
        # web3 Account._key_obj is an eth_keys.PrivateKey object
        # Account.key is a HexBytes, but _key_obj has the to_bytes() method
        private_key_bytes = context._raw_account._key_obj.to_bytes()

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
