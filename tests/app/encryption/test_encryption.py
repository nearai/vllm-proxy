"""
Tests for encryption and decryption functionality.
"""

import pytest
from app.encryption.encryption import (
    encrypt_data,
    decrypt_data,
)
from app.quote.quote import (
    ecdsa_context,
    ed25519_context,
    ECDSA,
    ED25519,
    SigningContext,
)


class TestECDSAEncryption:
    """Tests for ECDSA encryption and decryption."""

    def test_encrypt_decrypt_ecdsa_basic(self):
        """Test basic ECDSA encryption and decryption."""
        plaintext = b"Hello, World!"
        public_key = ecdsa_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ECDSA)
        assert encrypted is not None
        assert len(encrypted) > len(plaintext)  # Encrypted should be longer

        # Decrypt
        decrypted = decrypt_data(encrypted, ecdsa_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ecdsa_empty_data(self):
        """Test ECDSA encryption and decryption with empty data."""
        plaintext = b""
        public_key = ecdsa_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ECDSA)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ecdsa_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ecdsa_large_data(self):
        """Test ECDSA encryption and decryption with large data."""
        plaintext = b"x" * 10000  # 10KB of data
        public_key = ecdsa_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ECDSA)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ecdsa_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ecdsa_unicode(self):
        """Test ECDSA encryption and decryption with Unicode data."""
        plaintext = "Hello, 世界! 🌍".encode("utf-8")
        public_key = ecdsa_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ECDSA)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ecdsa_context)
        assert decrypted == plaintext
        assert decrypted.decode("utf-8") == "Hello, 世界! 🌍"

    def test_encrypt_decrypt_ecdsa_multiple_rounds(self):
        """Test multiple rounds of ECDSA encryption/decryption."""
        plaintext = b"Test message"
        public_key = ecdsa_context.signing_public_key

        # Multiple encrypt/decrypt cycles
        data = plaintext
        for _ in range(5):
            encrypted = encrypt_data(data, public_key, ECDSA)
            data = decrypt_data(encrypted, ecdsa_context)
            assert data == plaintext

    def test_encrypt_ecdsa_different_ciphertexts(self):
        """Test that ECDSA encryption produces different ciphertexts each time."""
        plaintext = b"Same plaintext"
        public_key = ecdsa_context.signing_public_key

        # Encrypt multiple times
        encrypted1 = encrypt_data(plaintext, public_key, ECDSA)
        encrypted2 = encrypt_data(plaintext, public_key, ECDSA)
        encrypted3 = encrypt_data(plaintext, public_key, ECDSA)

        # All should be different (due to ephemeral keys)
        assert encrypted1 != encrypted2
        assert encrypted2 != encrypted3
        assert encrypted1 != encrypted3

        # But all should decrypt to the same plaintext
        assert decrypt_data(encrypted1, ecdsa_context) == plaintext
        assert decrypt_data(encrypted2, ecdsa_context) == plaintext
        assert decrypt_data(encrypted3, ecdsa_context) == plaintext

    def test_encrypt_ecdsa_invalid_public_key(self):
        """Test ECDSA encryption with invalid public key."""
        plaintext = b"Test"
        invalid_key = "invalid_key_hex"

        with pytest.raises((ValueError, Exception)):
            encrypt_data(plaintext, invalid_key, ECDSA)

    def test_encrypt_ecdsa_wrong_length_public_key(self):
        """Test ECDSA encryption with wrong length public key."""
        plaintext = b"Test"
        # Too short
        short_key = "01" * 10  # 20 bytes instead of 64

        with pytest.raises(ValueError):
            encrypt_data(plaintext, short_key, ECDSA)

    def test_decrypt_ecdsa_invalid_data(self):
        """Test ECDSA decryption with invalid encrypted data."""
        invalid_data = b"too_short"

        with pytest.raises((ValueError, Exception)):
            decrypt_data(invalid_data, ecdsa_context)

    def test_decrypt_ecdsa_corrupted_data(self):
        """Test ECDSA decryption with corrupted encrypted data."""
        plaintext = b"Test message"
        public_key = ecdsa_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ECDSA)

        # Corrupt the data
        corrupted = bytearray(encrypted)
        corrupted[50] ^= 0xFF  # Flip some bits
        corrupted = bytes(corrupted)

        # Decryption should fail
        with pytest.raises(Exception):
            decrypt_data(corrupted, ecdsa_context)


class TestEd25519Encryption:
    """Tests for Ed25519 encryption and decryption."""

    def test_encrypt_decrypt_ed25519_basic(self):
        """Test basic Ed25519 encryption and decryption."""
        plaintext = b"Hello, World!"
        public_key = ed25519_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ED25519)
        assert encrypted is not None
        assert len(encrypted) > len(plaintext)  # Encrypted should be longer

        # Decrypt
        decrypted = decrypt_data(encrypted, ed25519_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ed25519_empty_data(self):
        """Test Ed25519 encryption and decryption with empty data."""
        plaintext = b""
        public_key = ed25519_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ED25519)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ed25519_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ed25519_large_data(self):
        """Test Ed25519 encryption and decryption with large data."""
        plaintext = b"x" * 10000  # 10KB of data
        public_key = ed25519_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ED25519)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ed25519_context)
        assert decrypted == plaintext

    def test_encrypt_decrypt_ed25519_unicode(self):
        """Test Ed25519 encryption and decryption with Unicode data."""
        plaintext = "Hello, 世界! 🌍".encode("utf-8")
        public_key = ed25519_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ED25519)
        assert encrypted is not None

        # Decrypt
        decrypted = decrypt_data(encrypted, ed25519_context)
        assert decrypted == plaintext
        assert decrypted.decode("utf-8") == "Hello, 世界! 🌍"

    def test_encrypt_decrypt_ed25519_multiple_rounds(self):
        """Test multiple rounds of Ed25519 encryption/decryption."""
        plaintext = b"Test message"
        public_key = ed25519_context.signing_public_key

        # Multiple encrypt/decrypt cycles
        data = plaintext
        for _ in range(5):
            encrypted = encrypt_data(data, public_key, ED25519)
            data = decrypt_data(encrypted, ed25519_context)
            assert data == plaintext

    def test_encrypt_ed25519_different_ciphertexts(self):
        """Test that Ed25519 encryption produces different ciphertexts each time."""
        plaintext = b"Same plaintext"
        public_key = ed25519_context.signing_public_key

        # Encrypt multiple times
        encrypted1 = encrypt_data(plaintext, public_key, ED25519)
        encrypted2 = encrypt_data(plaintext, public_key, ED25519)
        encrypted3 = encrypt_data(plaintext, public_key, ED25519)

        # All should be different (due to ephemeral keys)
        assert encrypted1 != encrypted2
        assert encrypted2 != encrypted3
        assert encrypted1 != encrypted3

        # But all should decrypt to the same plaintext
        assert decrypt_data(encrypted1, ed25519_context) == plaintext
        assert decrypt_data(encrypted2, ed25519_context) == plaintext
        assert decrypt_data(encrypted3, ed25519_context) == plaintext

    def test_encrypt_ed25519_invalid_public_key(self):
        """Test Ed25519 encryption with invalid public key."""
        plaintext = b"Test"
        invalid_key = "invalid_key_hex"

        with pytest.raises((ValueError, Exception)):
            encrypt_data(plaintext, invalid_key, ED25519)

    def test_encrypt_ed25519_wrong_length_public_key(self):
        """Test Ed25519 encryption with wrong length public key."""
        plaintext = b"Test"
        # Too short
        short_key = "01" * 10  # 20 bytes instead of 32

        with pytest.raises(ValueError):
            encrypt_data(plaintext, short_key, ED25519)

    def test_decrypt_ed25519_invalid_data(self):
        """Test Ed25519 decryption with invalid encrypted data."""
        # New format requires minimum 56 bytes: 32 (ephemeral public) + 24 (nonce) + 0 (ciphertext)
        invalid_data = b"too_short"  # Less than 56 bytes

        with pytest.raises((ValueError, Exception)):
            decrypt_data(invalid_data, ed25519_context)

    def test_decrypt_ed25519_corrupted_data(self):
        """Test Ed25519 decryption with corrupted encrypted data."""
        plaintext = b"Test message"
        public_key = ed25519_context.signing_public_key

        # Encrypt
        encrypted = encrypt_data(plaintext, public_key, ED25519)

        # Corrupt the data
        corrupted = bytearray(encrypted)
        corrupted[20] ^= 0xFF  # Flip some bits
        corrupted = bytes(corrupted)

        # Decryption should fail
        with pytest.raises(Exception):
            decrypt_data(corrupted, ed25519_context)


class TestCrossAlgorithm:
    """Tests for cross-algorithm scenarios."""

    def test_ecdsa_encrypted_cannot_decrypt_with_ed25519(self):
        """Test that ECDSA encrypted data cannot be decrypted with Ed25519 context."""
        plaintext = b"Test message"
        ecdsa_public_key = ecdsa_context.signing_public_key

        # Encrypt with ECDSA
        encrypted = encrypt_data(plaintext, ecdsa_public_key, ECDSA)

        # Try to decrypt with Ed25519 (should fail)
        with pytest.raises(Exception):
            decrypt_data(encrypted, ed25519_context)

    def test_ed25519_encrypted_cannot_decrypt_with_ecdsa(self):
        """Test that Ed25519 encrypted data cannot be decrypted with ECDSA context."""
        plaintext = b"Test message"
        ed25519_public_key = ed25519_context.signing_public_key

        # Encrypt with Ed25519
        encrypted = encrypt_data(plaintext, ed25519_public_key, ED25519)

        # Try to decrypt with ECDSA (should fail)
        with pytest.raises(Exception):
            decrypt_data(encrypted, ecdsa_context)

    def test_encrypt_with_wrong_algorithm(self):
        """Test encryption with unsupported algorithm."""
        plaintext = b"Test"
        public_key = ecdsa_context.signing_public_key

        with pytest.raises(ValueError):
            encrypt_data(plaintext, public_key, "invalid_algo")

    def test_decrypt_with_wrong_algorithm(self):
        """Test decryption with unsupported algorithm in context."""
        # Create a context with invalid method
        invalid_context = SigningContext(
            method="invalid",
            signing_address="0x123",
            signing_address_bytes=b"\x01" * 20,
            signing_public_key="01" * 32,
        )

        plaintext = b"Test"
        public_key = ecdsa_context.signing_public_key
        encrypted = encrypt_data(plaintext, public_key, ECDSA)

        with pytest.raises(ValueError):
            decrypt_data(encrypted, invalid_context)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_encrypt_decrypt_single_byte(self):
        """Test encryption/decryption of a single byte."""
        plaintext = b"x"
        ecdsa_public_key = ecdsa_context.signing_public_key
        ed25519_public_key = ed25519_context.signing_public_key

        # ECDSA
        encrypted = encrypt_data(plaintext, ecdsa_public_key, ECDSA)
        assert decrypt_data(encrypted, ecdsa_context) == plaintext

        # Ed25519
        encrypted = encrypt_data(plaintext, ed25519_public_key, ED25519)
        assert decrypt_data(encrypted, ed25519_context) == plaintext

    def test_encrypt_decrypt_binary_data(self):
        """Test encryption/decryption of binary data (null bytes, etc.)."""
        plaintext = bytes(range(256))  # All possible byte values
        ecdsa_public_key = ecdsa_context.signing_public_key
        ed25519_public_key = ed25519_context.signing_public_key

        # ECDSA
        encrypted = encrypt_data(plaintext, ecdsa_public_key, ECDSA)
        assert decrypt_data(encrypted, ecdsa_context) == plaintext

        # Ed25519
        encrypted = encrypt_data(plaintext, ed25519_public_key, ED25519)
        assert decrypt_data(encrypted, ed25519_context) == plaintext

    def test_encrypt_decrypt_repeated_patterns(self):
        """Test encryption/decryption of data with repeated patterns."""
        plaintext = b"AAAA" * 1000  # Repeated pattern
        ecdsa_public_key = ecdsa_context.signing_public_key
        ed25519_public_key = ed25519_context.signing_public_key

        # ECDSA
        encrypted = encrypt_data(plaintext, ecdsa_public_key, ECDSA)
        assert decrypt_data(encrypted, ecdsa_context) == plaintext

        # Ed25519
        encrypted = encrypt_data(plaintext, ed25519_public_key, ED25519)
        assert decrypt_data(encrypted, ed25519_context) == plaintext
