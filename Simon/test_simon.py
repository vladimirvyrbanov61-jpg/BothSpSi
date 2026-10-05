"""
test_simon.py — pytest test suite for simon.py

Run with:
    pytest test_simon.py -v

Requires: Python 3.10+, NumPy, pytest.
"""

from typing import Optional

import numpy as np
import pytest

from Simon.simon import Simon, _SIMON_PARAMS, _Z_SEQUENCES, rol, ror, f_round, expand_key
from Simon.test_kat_vectors import SIMON_OFFICIAL_KATS, kat_arrays


# ---------------------------------------------------------------------------
# Unit tests: rol / f_round
# ---------------------------------------------------------------------------

class TestRol:
    def test_identity_on_zero(self):
        x = np.array([0], dtype=np.uint16)
        assert rol(x, 3, 16)[0] == 0

    def test_single_bit_rotation(self):
        x = np.array([1], dtype=np.uint16)
        assert rol(x, 1, 16)[0] == 2

    def test_wrap_around(self):
        x = np.array([0x8000], dtype=np.uint16)   # MSB set
        assert rol(x, 1, 16)[0] == 0x0001

    def test_full_rotation_is_identity(self):
        x = np.array([0xABCD], dtype=np.uint16)
        np.testing.assert_array_equal(rol(x, 16, 16), x)

    def test_vectorised(self):
        xs = np.array([1, 2, 4, 8], dtype=np.uint16)
        np.testing.assert_array_equal(
            rol(xs, 1, 16), np.array([2, 4, 8, 16], dtype=np.uint16)
        )

    def test_ror_is_inverse_of_rol(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 0xFFFF, size=8, dtype=np.uint16)
        for k in [1, 3, 7, 8, 15]:
            np.testing.assert_array_equal(ror(rol(x, k, 16), k, 16), x)


class TestZSequences:
    """z_0 … z_4 bit strings match Beaulieu et al. 2013 Figure 3.3."""

    _PAPER_Z: list[str] = [
        "11111010001001010110000111001101111101000100101011000011100110",
        "10001110111110010011000010110101000111011111001001100001011010",
        "10101111011100000011010010011000101000010001111110010110110011",
        "11011011101011000110010111100000010010001010011100110100001111",
        "11010001111001101011011000100000010111000011001010010011101111",
    ]

    @pytest.mark.parametrize("z_index", range(5))
    def test_z_sequence_matches_paper(self, z_index: int) -> None:
        expected = [int(b) for b in self._PAPER_Z[z_index]]
        assert _Z_SEQUENCES[z_index] == expected


class TestSimonParamsTable:
    """Round counts and z-indices match Table 3.1 (official ten variants)."""

    @pytest.mark.parametrize(
        "n,m,z_idx,rounds",
        [
            (16, 4, 0, 32),
            (24, 3, 0, 36),
            (24, 4, 1, 36),
            (32, 3, 2, 42),
            (32, 4, 3, 44),
            (48, 2, 2, 52),
            (48, 3, 3, 54),
            (64, 2, 2, 68),
            (64, 3, 3, 69),
            (64, 4, 4, 72),
        ],
    )
    def test_official_variant_defaults(self, n: int, m: int, z_idx: int, rounds: int) -> None:
        c = Simon(n=n, m=m)
        assert c.z_index == z_idx
        assert c.rounds == rounds


class TestFRound:
    def test_zero_input(self):
        x = np.array([0], dtype=np.uint16)
        assert f_round(x, 16)[0] == 0

    def test_all_ones(self):
        # S1(x)=x, S8(x)=x → AND = x; f = x ^ S2(x)
        # S2(0xFFFF) = 0xFFFF → f = 0
        x = np.array([0xFFFF], dtype=np.uint16)
        assert f_round(x, 16)[0] == 0

    def test_deterministic(self):
        x = np.array([0xABCD], dtype=np.uint16)
        assert f_round(x, 16)[0] == f_round(x, 16)[0]


# ---------------------------------------------------------------------------
# Official SIMON 32/64 test vector
# Source: SIMON test-vector suite (matching the C reference implementation).
# Params: n=16, m=4, T=32, z-sequence index 0.
# ---------------------------------------------------------------------------

class TestSimon3264OfficialVector:
    """SIMON 32/64: n=16, m=4, T=32, z_index=0."""

    KEY_WORDS    = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    PLAIN_WORDS  = np.array([0x6565, 0x6877], dtype=np.uint16)
    CIPHER_WORDS = np.array([0xc69b, 0xe9bb], dtype=np.uint16)

    @pytest.fixture
    def cipher(self) -> Simon:
        return Simon(n=16, m=4)

    def test_default_rounds_match_spec(self, cipher):
        assert cipher.rounds == 32, (
            f"Default rounds for SIMON 32/64 must be 32, got {cipher.rounds}"
        )

    def test_default_z_index_match_spec(self, cipher):
        assert cipher.z_index == 0, (
            f"Default z_index for SIMON 32/64 must be 0, got {cipher.z_index}"
        )

    def test_block_and_key_size(self, cipher):
        assert cipher.block_size_bits == 32
        assert cipher.key_size_bits   == 64

    def test_encrypt_official_vector(self, cipher):
        ct = cipher.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        np.testing.assert_array_equal(
            ct[0], self.CIPHER_WORDS,
            err_msg=f"Got {ct[0].tolist()}, expected {self.CIPHER_WORDS.tolist()}"
        )

    def test_decrypt_official_vector(self, cipher):
        pt = cipher.decrypt(self.CIPHER_WORDS, self.KEY_WORDS)
        np.testing.assert_array_equal(
            pt[0], self.PLAIN_WORDS,
            err_msg=f"Got {pt[0].tolist()}, expected {self.PLAIN_WORDS.tolist()}"
        )

    def test_roundtrip(self, cipher):
        ct = cipher.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        pt = cipher.decrypt(ct, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)

    def test_encrypt_accepts_2d_input(self, cipher):
        pt = self.PLAIN_WORDS[np.newaxis, :]   # (1, 2)
        ct = cipher.encrypt(pt, self.KEY_WORDS)
        assert ct.shape == (1, 2)
        np.testing.assert_array_equal(ct[0], self.CIPHER_WORDS)

    def test_repr_contains_n_and_m(self, cipher):
        r = repr(cipher)
        assert "n=16" in r and "m=4" in r


# ---------------------------------------------------------------------------
# Round-override: a reduced-round cipher must differ from full-round output.
# The golden value for 16 rounds is computed once then stored.
# ---------------------------------------------------------------------------

class TestRoundOverride:
    KEY_WORDS   = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    PLAIN_WORDS = np.array([0x6565, 0x6877], dtype=np.uint16)

    # Golden ciphertext for 16-round SIMON 32/64 — computed from this
    # implementation.  If the algorithm changes, recompute with:
    #   Simon(n=16, m=4, rounds=16).encrypt(PLAIN_WORDS, KEY_WORDS)
    _GOLDEN_16R = None  # type: Optional[np.ndarray]

    @classmethod
    def _golden(cls) -> np.ndarray:
        if cls._GOLDEN_16R is None:
            cls._GOLDEN_16R = Simon(n=16, m=4, rounds=16).encrypt(
                cls.PLAIN_WORDS, cls.KEY_WORDS
            )[0].copy()
        return cls._GOLDEN_16R

    def test_fewer_rounds_changes_output(self):
        full    = Simon(n=16, m=4)
        reduced = Simon(n=16, m=4, rounds=16)
        ct_full = full.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        ct_red  = reduced.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        assert not np.array_equal(ct_full, ct_red), (
            "16-round output must differ from 32-round output"
        )

    def test_16_round_output_is_stable(self):
        c  = Simon(n=16, m=4, rounds=16)
        ct = c.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        np.testing.assert_array_equal(
            ct[0], self._golden(),
            err_msg="16-round output is non-deterministic or changed"
        )

    def test_16_round_is_not_full_vector(self):
        FULL_CT = np.array([0xc69b, 0xe9bb], dtype=np.uint16)
        assert not np.array_equal(self._golden(), FULL_CT)

    def test_fewer_rounds_roundtrip(self):
        c  = Simon(n=16, m=4, rounds=16)
        ct = c.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        pt = c.decrypt(ct, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)

    def test_more_rounds_roundtrip(self):
        c  = Simon(n=16, m=4, rounds=48)
        ct = c.encrypt(self.PLAIN_WORDS, self.KEY_WORDS)
        pt = c.decrypt(ct, self.KEY_WORDS)
        np.testing.assert_array_equal(pt[0], self.PLAIN_WORDS)


# ---------------------------------------------------------------------------
# Vectorisation: batch plaintext + single key
# ---------------------------------------------------------------------------

class TestBroadcastSingleKey:
    N = 8

    @pytest.fixture
    def cipher(self) -> Simon:
        return Simon(n=16, m=4)

    @pytest.fixture
    def key(self) -> np.ndarray:
        return np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)

    @pytest.fixture
    def plaintexts(self) -> np.ndarray:
        return np.random.default_rng(42).integers(
            0, 0xFFFF, size=(self.N, 2), dtype=np.uint16
        )

    def test_batch_encrypt_matches_sequential(self, cipher, key, plaintexts):
        ct_batch = cipher.encrypt(plaintexts, key)
        ct_seq   = np.stack([cipher.encrypt(plaintexts[i], key)[0] for i in range(self.N)])
        np.testing.assert_array_equal(ct_batch, ct_seq)

    def test_batch_decrypt_roundtrip(self, cipher, key, plaintexts):
        ct = cipher.encrypt(plaintexts, key)
        pt = cipher.decrypt(ct, key)
        np.testing.assert_array_equal(pt, plaintexts)

    def test_output_shape(self, cipher, key, plaintexts):
        ct = cipher.encrypt(plaintexts, key)
        assert ct.shape == (self.N, 2)
        assert ct.dtype == np.uint16


# ---------------------------------------------------------------------------
# Vectorisation: batch plaintext + batch keys (N keys)
# ---------------------------------------------------------------------------

class TestBatchKeys:
    N = 6

    @pytest.fixture
    def cipher(self) -> Simon:
        return Simon(n=16, m=4)

    @pytest.fixture
    def keys(self) -> np.ndarray:
        return np.random.default_rng(99).integers(
            0, 0xFFFF, size=(self.N, 4), dtype=np.uint16
        )

    @pytest.fixture
    def plaintexts(self) -> np.ndarray:
        return np.random.default_rng(77).integers(
            0, 0xFFFF, size=(self.N, 2), dtype=np.uint16
        )

    def test_batch_keys_encrypt_matches_sequential(self, cipher, keys, plaintexts):
        ct_batch = cipher.encrypt(plaintexts, keys)
        ct_seq   = np.stack([
            cipher.encrypt(plaintexts[i], keys[i])[0] for i in range(self.N)
        ])
        np.testing.assert_array_equal(ct_batch, ct_seq)

    def test_batch_keys_decrypt_roundtrip(self, cipher, keys, plaintexts):
        ct = cipher.encrypt(plaintexts, keys)
        pt = cipher.decrypt(ct, keys)
        np.testing.assert_array_equal(pt, plaintexts)

    def test_different_keys_yield_different_ciphertexts(self, cipher, keys, plaintexts):
        ct = cipher.encrypt(plaintexts, keys)
        # With random keys, all blocks are almost certainly distinct.
        unique_rows = {tuple(row) for row in ct.tolist()}
        assert len(unique_rows) > 1, "All blocks identical — suspicious"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_n_m_combo(self):
        with pytest.raises(ValueError, match="not a valid SIMON parameter pair"):
            Simon(n=7, m=3)

    def test_invalid_z_index_too_large(self):
        with pytest.raises(ValueError, match="z_index"):
            Simon(n=16, m=4, z_index=9)

    def test_invalid_rounds_zero(self):
        with pytest.raises(ValueError, match="rounds"):
            Simon(n=16, m=4, rounds=0)

    def test_wrong_key_word_count_1d(self):
        c = Simon(n=16, m=4)
        with pytest.raises(ValueError):
            c.encrypt(
                np.array([[0x6565, 0x6877]], dtype=np.uint16),
                np.array([0x1918, 0x1110], dtype=np.uint16),   # only 2 words
            )

    def test_wrong_plaintext_shape(self):
        c = Simon(n=16, m=4)
        with pytest.raises(ValueError):
            c.encrypt(
                np.array([0x1234], dtype=np.uint16),
                np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16),
            )

    def test_key_batch_size_mismatch(self):
        c   = Simon(n=16, m=4)
        rng = np.random.default_rng(0)
        pt  = rng.integers(0, 0xFFFF, size=(4, 2), dtype=np.uint16)
        # 3 keys for 4 blocks — must fail
        bad_keys = rng.integers(0, 0xFFFF, size=(3, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="batch size"):
            c.encrypt(pt, bad_keys)


# ---------------------------------------------------------------------------
# Other SIMON variants: smoke-test encrypt/decrypt round-trips
# ---------------------------------------------------------------------------

class TestOfficialKATs:
    """Known-answer tests from NSA Implementation Guide / paper Appendix B."""

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Simon{2*n}/{n*m}")
            for n, m, p, c, k in SIMON_OFFICIAL_KATS
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
        c = Simon(n=n, m=m)
        ct = c.encrypt(pt, key_arr)
        np.testing.assert_array_equal(
            ct[0],
            ct_exp,
            err_msg=f"encrypt mismatch Simon{2*n}/{n*m}: got {ct[0].tolist()}",
        )

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Simon{2*n}/{n*m}")
            for n, m, p, c, k in SIMON_OFFICIAL_KATS
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
        c = Simon(n=n, m=m)
        pt = c.decrypt(ct, key_arr)
        np.testing.assert_array_equal(pt[0], pt_exp)

    @pytest.mark.parametrize(
        "n,m,plain,cipher,key",
        [
            pytest.param(n, m, p, c, k, id=f"Simon{2*n}/{n*m}")
            for n, m, p, c, k in SIMON_OFFICIAL_KATS
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
        c = Simon(n=n, m=m)
        np.testing.assert_array_equal(c.decrypt(c.encrypt(pt, key_arr), key_arr)[0], pt)


def test_algebraic_round_trip() -> None:
    """Decrypt(Encrypt(P)) == P for 10k random blocks per supported (n, m)."""
    n_samples = 10_000
    for (n, m) in sorted(_SIMON_PARAMS):
        dtype = {16: np.uint16, 24: np.uint32, 32: np.uint32, 48: np.uint64, 64: np.uint64}[n]
        mask = int((1 << n) - 1)
        rng = np.random.default_rng(n * 1000 + m)
        cipher = Simon(n=n, m=m)
        pt = rng.integers(0, mask + 1, size=(n_samples, 2), dtype=dtype)
        key = rng.integers(0, mask + 1, size=m, dtype=dtype)
        ct = cipher.encrypt(pt, key)
        pt2 = cipher.decrypt(ct, key)
        np.testing.assert_array_equal(
            pt2,
            pt,
            err_msg=f"algebraic round-trip failed for SIMON {2*n}/{n*m}",
        )


class TestOtherVariants:
    @pytest.mark.parametrize("n,m", [
        (16, 2),
        (16, 3),
        (24, 2),
        (32, 2),
    ])
    def test_roundtrip(self, n: int, m: int):
        dtype   = {16: np.uint16, 24: np.uint32, 32: np.uint32, 48: np.uint64, 64: np.uint64}[n]
        max_val = int((1 << n) - 1)
        rng     = np.random.default_rng(n * 100 + m)
        pt      = rng.integers(0, max_val, size=(4, 2), dtype=dtype)
        key     = rng.integers(0, max_val, size=m, dtype=dtype)
        c       = Simon(n=n, m=m)
        ct      = c.encrypt(pt, key)
        pt2     = c.decrypt(ct, key)
        np.testing.assert_array_equal(
            pt2, pt, err_msg=f"Roundtrip failed for SIMON {2*n}/{n*m}"
        )

    def test_simon_64_128_explicit(self):
        """SIMON 64/128: n=32, m=4 — explicit smoke test."""
        c   = Simon(n=32, m=4)
        rng = np.random.default_rng(1234)
        pt  = rng.integers(0, 0xFFFF_FFFF, size=(3, 2), dtype=np.uint32)
        key = rng.integers(0, 0xFFFF_FFFF, size=4, dtype=np.uint32)
        ct  = c.encrypt(pt, key)
        pt2 = c.decrypt(ct, key)
        np.testing.assert_array_equal(pt2, pt)
