"""Tests for speck.py — SPECK family primitive."""

from __future__ import annotations

import numpy as np
import pytest

from Speck.speck import Speck, _SPECK_PARAMS, decrypt_blocks, encrypt_blocks, expand_key, rol, ror
from Speck.test_kat_vectors import SPECK_OFFICIAL_KATS, kat_arrays


class TestSpeckParamsTable:
    """α, β, and round counts match Beaulieu et al. 2013 Table 4.1."""

    @pytest.mark.parametrize(
        "n,m,alpha,beta,rounds",
        [
            (16, 4, 7, 2, 22),
            (24, 3, 8, 3, 22),
            (24, 4, 8, 3, 23),
            (32, 3, 8, 3, 26),
            (32, 4, 8, 3, 27),
            (48, 2, 8, 3, 28),
            (48, 3, 8, 3, 29),
            (64, 2, 8, 3, 32),
            (64, 3, 8, 3, 33),
            (64, 4, 8, 3, 34),
        ],
    )
    def test_official_variant_defaults(
        self, n: int, m: int, alpha: int, beta: int, rounds: int
    ) -> None:
        c = Speck(n=n, m=m)
        assert c.alpha == alpha
        assert c.beta == beta
        assert c.rounds == rounds

    def test_invalid_round_and_rotation_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            Speck(n=16, m=4, rounds=0)
        with pytest.raises(ValueError, match="alpha"):
            Speck(n=16, m=4, alpha=0)


class TestRotations:
    def test_rol_ror_inverse_16bit(self):
        x = np.array([0xABCD], dtype=np.uint16)
        assert ror(rol(x, 7, 16), 7, 16)[0] == x[0]
        assert rol(ror(x, 7, 16), 7, 16)[0] == x[0]


class TestSpeck3264OfficialVector:
    """SPECK 32/64: n=16, m=4, T=22, α=7, β=2.

    Reference: NSA / community test vector (little-endian byte order).
    Plaintext bytes 4c 69 74 65 → words x=0x6574, y=0x694c in [left, right] layout.
  Key words [l2, l1, l0, k0] = [0x1918, 0x1110, 0x0908, 0x0100].
    """

    KEY_WORDS = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    PLAIN_WORDS = np.array([0x6574, 0x694C], dtype=np.uint16)
    CIPHER_WORDS = np.array([0xA868, 0x42F2], dtype=np.uint16)

    @pytest.fixture
    def cipher(self) -> Speck:
        return Speck(n=16, m=4)

    def test_default_rounds(self, cipher: Speck) -> None:
        assert cipher.rounds == 22
        assert cipher.alpha == 7
        assert cipher.beta == 2

    def test_encrypt_official_vector(self, cipher: Speck) -> None:
        ct = cipher.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        np.testing.assert_array_equal(ct[0], self.CIPHER_WORDS)

    def test_decrypt_official_vector(self, cipher: Speck) -> None:
        pt = cipher.decrypt(self.CIPHER_WORDS, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)

    def test_roundtrip(self, cipher: Speck) -> None:
        ct = cipher.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        pt = cipher.decrypt(ct, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)

    def test_encrypt_blocks_matches_class(self, cipher: Speck) -> None:
        pt = self.PLAIN_WORDS[np.newaxis, :]
        sk = expand_key(self.KEY_WORDS[np.newaxis, :], 16, 4, 22, 7, 2)
        ct = encrypt_blocks(pt, sk, 16, 22, 7, 2)
        np.testing.assert_array_equal(ct[0], self.CIPHER_WORDS)


class TestOfficialKATs:
    """Known-answer tests from NSA Implementation Guide / paper Appendix C."""

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Speck{2*n}/{n*m}")
            for n, m, p, c, k in SPECK_OFFICIAL_KATS
        ],
    )
    def test_encrypt_official_vector(
        self,
        n: int,
        m: int,
        plain: list[int],
        cipher: list[int],
        key: list[int],
    ) -> None:
        pt, ct_exp, key_arr = kat_arrays(n, plain, cipher, key)
        c = Speck(n=n, m=m)
        ct = c.encrypt(pt, key_arr)
        np.testing.assert_array_equal(ct[0], ct_exp)

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Speck{2*n}/{n*m}")
            for n, m, p, c, k in SPECK_OFFICIAL_KATS
        ],
    )
    def test_decrypt_official_vector(
        self,
        n: int,
        m: int,
        plain: list[int],
        cipher: list[int],
        key: list[int],
    ) -> None:
        pt_exp, ct, key_arr = kat_arrays(n, plain, cipher, key)
        c = Speck(n=n, m=m)
        pt = c.decrypt(ct, key_arr)
        np.testing.assert_array_equal(pt[0], pt_exp)

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Speck{2*n}/{n*m}")
            for n, m, p, c, k in SPECK_OFFICIAL_KATS
        ],
    )
    def test_kat_roundtrip(
        self,
        n: int,
        m: int,
        plain: list[int],
        cipher: list[int],
        key: list[int],
    ) -> None:
        pt, ct, key_arr = kat_arrays(n, plain, cipher, key)
        c = Speck(n=n, m=m)
        np.testing.assert_array_equal(c.decrypt(c.encrypt(pt, key_arr), key_arr)[0], pt)


def test_algebraic_round_trip() -> None:
    """Decrypt(Encrypt(P)) == P for 10k random blocks per supported (n, m)."""
    n_samples = 10_000
    for (n, m) in sorted(_SPECK_PARAMS):
        dtype = {16: np.uint16, 24: np.uint32, 32: np.uint32, 48: np.uint64, 64: np.uint64}[n]
        mask = int((1 << n) - 1)
        rng = np.random.default_rng(n * 2000 + m)
        cipher = Speck(n=n, m=m)
        pt = rng.integers(0, mask + 1, size=(n_samples, 2), dtype=dtype)
        key = rng.integers(0, mask + 1, size=m, dtype=dtype)
        ct = cipher.encrypt(pt, key)
        pt2 = cipher.decrypt(ct, key)
        np.testing.assert_array_equal(
            pt2,
            pt,
            err_msg=f"algebraic round-trip failed for SPECK {2*n}/{n*m}",
        )


class TestRoundOverride:
    KEY_WORDS = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    PLAIN_WORDS = np.array([0x6574, 0x694C], dtype=np.uint16)

    def test_fewer_rounds_changes_output(self) -> None:
        full = Speck(n=16, m=4)
        reduced = Speck(n=16, m=4, rounds=11)
        ct_full = full.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        ct_red = reduced.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        assert not np.array_equal(ct_full, ct_red)

    def test_partial_roundtrip(self) -> None:
        c = Speck(n=16, m=4, rounds=11)
        ct = c.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        pt = c.decrypt(ct, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)
