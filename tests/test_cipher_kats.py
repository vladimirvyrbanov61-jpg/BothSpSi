"""Known-answer tests for the active SIMON32/64 and SPECK32/64 profiles."""

from __future__ import annotations

import numpy as np

from ciphers.registry import get_cipher


def test_simon3264_official_known_answer() -> None:
    cipher = get_cipher("simon")
    plaintext = np.array([0x6565, 0x6877], dtype=np.uint16)
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    expected = np.array([0xC69B, 0xE9BB], dtype=np.uint16)

    np.testing.assert_array_equal(cipher.encrypt(plaintext, key)[0], expected)


def test_speck3264_official_known_answer() -> None:
    cipher = get_cipher("speck")
    plaintext = np.array([0x6574, 0x694C], dtype=np.uint16)
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    expected = np.array([0xA868, 0x42F2], dtype=np.uint16)

    np.testing.assert_array_equal(cipher.encrypt(plaintext, key)[0], expected)
