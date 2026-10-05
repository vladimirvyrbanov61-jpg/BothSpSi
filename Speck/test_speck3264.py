"""Tests for speck3264 profile layer."""

from __future__ import annotations

import numpy as np
import pytest

from Speck.speck3264 import ROUNDS, Speck3264
from Speck.speck3264.cipher import ALPHA, BETA
from Speck.test_speck import TestSpeck3264OfficialVector


class TestSpeck3264Cipher:
    def test_constants(self) -> None:
        assert ROUNDS == 22
        assert ALPHA == 7
        assert BETA == 2

    def test_profile_encrypt_matches_primitive(self) -> None:
        key = TestSpeck3264OfficialVector.KEY_WORDS
        pt = TestSpeck3264OfficialVector.PLAIN_WORDS
        expected = TestSpeck3264OfficialVector.CIPHER_WORDS
        cipher = Speck3264()
        ct = cipher.encrypt(pt, key)
        np.testing.assert_array_equal(ct[0], expected)

    def test_encrypt_rounds_partial(self) -> None:
        cipher = Speck3264()
        key = TestSpeck3264OfficialVector.KEY_WORDS
        pt = TestSpeck3264OfficialVector.PLAIN_WORDS[np.newaxis, :]
        ct11 = cipher.encrypt_rounds(pt, key, 11)
        ct22 = cipher.encrypt_rounds(pt, key, 22)
        assert not np.array_equal(ct11[0], ct22[0])

    def test_encrypt_rounds_bounds(self) -> None:
        cipher = Speck3264()
        with pytest.raises(ValueError):
            cipher.encrypt_rounds(
                TestSpeck3264OfficialVector.PLAIN_WORDS[np.newaxis, :],
                TestSpeck3264OfficialVector.KEY_WORDS,
                0,
            )
