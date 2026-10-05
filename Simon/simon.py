"""
simon.py — SIMON block cipher (ECB-mode raw block functions only).

⚠️  PURPOSE: Research / benchmarking / education only.
    NOT for production cryptographic use.
    No constant-time guarantees; no side-channel mitigations.

Supported family
----------------
SIMON 2n / mn for any (n, m) pair listed in the spec.
Default: Simon(n=16, m=4) → SIMON 32/64, 32 rounds, z-sequence j=0.

References
----------
Beaulieu et al., "The SIMON and SPECK Families of Lightweight Block Ciphers",
NSA / IACR Technical Report, 2013.  https://eprint.iacr.org/2013/404.pdf

Run tests (requires pytest):
    pytest test_simon.py -v

Public API
----------
Simon(n, m, *, z_index=None, rounds=None)
    .encrypt(plaintext, key) → ciphertext
    .decrypt(ciphertext, key) → plaintext

Array shapes / word ordering
------------------------------
All functions consume and return *word arrays*, not byte strings.

plaintext / ciphertext : ndarray, shape (N, 2), dtype from n
    axis 0 : batch dimension N (number of blocks).
    axis 1 : [left_word, right_word]
              left_word  = "x" half — the half that the round function f acts on.
              right_word = "y" half.

key : ndarray, shape (N, m) | (1, m) | (m,)
    m words in big-endian key-word order:
        key[..., 0] = k_{m-1}  (most-significant key word)
        key[..., m-1] = k_0   (least-significant key word)
    This matches the convention used in the SIMON test vectors (key listed
    with the most-significant word first).  Internally the schedule reverses
    the order to k_0 … k_{m-1} before expansion.
    A shape-(m,) or (1, m) key broadcasts across all N blocks.
    Shape (N, m) supplies an independent key per block.

dtype policy
------------
    n=16 → uint16,  n=32 → uint32,  n=64 → uint64.
    Intermediate values ≤ n bits are always masked to n bits.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# SIMON specification tables (Beaulieu et al. 2013, Table 3 / Algorithm 2)
# ---------------------------------------------------------------------------

# Five z-constant LFSR sequences (period 62), Beaulieu et al. 2013 Figure 3.3.
# Stored as integers with paper bits reversed (LSB = paper z[0]), matching reference code.
_Z_BITS: list[int] = [
    int("".join(reversed(s)), 2)
    for s in (
        "11111010001001010110000111001101111101000100101011000011100110",
        "10001110111110010011000010110101000111011111001001100001011010",
        "10101111011100000011010010011000101000010001111110010110110011",
        "11011011101011000110010111100000010010001010011100110100001111",
        "11010001111001101011011000100000010111000011001010010011101111",
    )
]
# Expose list form for tests (index i = paper z[i]).
_Z_SEQUENCES: list[list[int]] = [
    [int((z >> i) & 1) for i in range(62)] for z in _Z_BITS
]

# (n, m) → (z_index, default_rounds).
# Official ten variants: Beaulieu et al. 2013 Table 3.1 / Appendix B.
# Extra (n, m) pairs are research extensions (algebraic tests only).
_SIMON_PARAMS: dict[tuple[int, int], tuple[int, int]] = {
    # --- Official family (block bits / key bits) ---
    (16, 4): (0, 32),    # SIMON 32/64
    (24, 3): (0, 36),    # SIMON 48/72
    (24, 4): (1, 36),    # SIMON 48/96
    (32, 3): (2, 42),    # SIMON 64/96
    (32, 4): (3, 44),    # SIMON 64/128
    (48, 2): (2, 52),    # SIMON 96/96
    (48, 3): (3, 54),    # SIMON 96/144
    (64, 2): (2, 68),    # SIMON 128/128
    (64, 3): (3, 69),    # SIMON 128/192
    (64, 4): (4, 72),    # SIMON 128/256 — z_4 per Table 3.1
    # --- Non-standard extensions (no Appendix B vector) ---
    (16, 2): (0, 32),
    (16, 3): (0, 36),
    (24, 2): (0, 36),
    (32, 2): (1, 32),
}

# ---------------------------------------------------------------------------
# dtype helper
# ---------------------------------------------------------------------------

def _dtype_for_n(n: int) -> np.dtype:
    """Return the smallest unsigned NumPy dtype that holds an n-bit word."""
    for bits, dt in [(8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)]:
        if n <= bits:
            return np.dtype(dt)
    raise ValueError(f"n={n} exceeds 64-bit word width")


# ---------------------------------------------------------------------------
# Core bit operations — pure, vectorised
# ---------------------------------------------------------------------------

def rol(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """Left-rotate each element of *x* by *k* bits within an n-bit word.

    Parameters
    ----------
    x : ndarray of unsigned integers
    k : rotation amount (0 < k < n)
    n : word width in bits

    Returns
    -------
    ndarray — same shape and dtype as *x*.
    """
    k = k % n
    if k == 0:
        return x.copy()
    dt   = x.dtype
    mask = dt.type((1 << n) - 1)
    return ((x << dt.type(k)) | (x >> dt.type(n - k))) & mask


def ror(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """Right-rotate each element of *x* by *k* bits within an n-bit word."""
    return rol(x, n - (k % n), n)


def f_round(x: np.ndarray, n: int) -> np.ndarray:
    """SIMON round function  f(x) = (S¹x & S⁸x) ⊕ S²x.

    S^k denotes *left* rotation by k bits.

    Parameters
    ----------
    x : ndarray of unsigned integers, any batch shape
    n : word width in bits

    Returns
    -------
    ndarray — same shape and dtype as *x*.
    """
    return (rol(x, 1, n) & rol(x, 8, n)) ^ rol(x, 2, n)


# ---------------------------------------------------------------------------
# Key schedule
# ---------------------------------------------------------------------------

def expand_key(
    key_words: np.ndarray,
    n: int,
    m: int,
    rounds: int,
    z_index: int,
) -> np.ndarray:
    """Expand key words into *rounds* subkeys (Algorithm 2 from the paper).

    Parameters
    ----------
    key_words : ndarray, shape (..., m)
        Key in big-endian word order: [..., k_{m-1}, …, k_0].
        Leading dimensions are the batch axis.
    n         : word width in bits
    m         : number of key words
    rounds    : total round count T
    z_index   : z-sequence index (0–4)

    Returns
    -------
    ndarray, shape (..., rounds)
        Subkey array; K[..., i] is used in encryption round i.
    """
    dtype  = _dtype_for_n(n)
    mask   = dtype.type((1 << n) - 1)
    z_int  = _Z_BITS[z_index]
    period = 62

    # Reverse to ascending (k_0 first) for the schedule recurrence.
    key_asc = key_words[..., ::-1].astype(dtype)  # (..., m)

    # Build a flat list of subkey arrays, each of shape batch_shape.
    K: list[np.ndarray] = [key_asc[..., i] for i in range(m)]

    # c = 2^n − 4 = all-ones ^ 3
    c = mask ^ dtype.type(3)

    for i in range(m, rounds):
        # K_i = c ⊕ z[(i−m) mod 62] ⊕ K_{i−m} ⊕ (I ⊕ S^{−1})(S^{−3}(K_{i−1}) [⊕ K_{i−3} if m≥4])
        tmp = ror(K[i - 1], 3, n)
        if m == 4:
            tmp = tmp ^ K[i - 3]
        tmp   = tmp ^ ror(tmp, 1, n)
        zi    = dtype.type((z_int >> ((i - m) % period)) & 1)
        K.append((c ^ zi ^ K[i - m] ^ tmp) & mask)

    return np.stack(K, axis=-1)   # (..., rounds)


# ---------------------------------------------------------------------------
# Encryption / decryption — vectorised over N blocks
# ---------------------------------------------------------------------------

def _enc_round(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One SIMON encryption round.

    SIMON Feistel update (both halves are n-bit words):
        x_new = y ⊕ f(x) ⊕ k
        y_new = x

    Parameters
    ----------
    x, y : ndarray, shape (N,) — left and right half-words.
    k    : ndarray, shape (N,) or (1,) — round subkey (NumPy broadcasts).
    n    : word width in bits.
    """
    mask = x.dtype.type((1 << n) - 1)
    return (y ^ f_round(x, n) ^ k) & mask, x


def _dec_round(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One SIMON decryption round (exact inverse of _enc_round).

    Derivation — given output (x', y') of one encryption round where
        x' = y ⊕ f(x) ⊕ k,   y' = x
    recover the inputs:
        x_prev = y' = y,   y_prev = x' ⊕ f(y') ⊕ k = x' ⊕ f(x) ⊕ k = y
    More concretely:
        x = y',              y = x' ⊕ f(y') ⊕ k
    """
    mask = x.dtype.type((1 << n) - 1)
    return y, (x ^ f_round(y, n) ^ k) & mask


def encrypt_blocks(
    plaintext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
) -> np.ndarray:
    """Encrypt N plaintext blocks using pre-expanded subkeys.

    Parameters
    ----------
    plaintext : ndarray, shape (N, 2), dtype compatible with n
        plaintext[:, 0] = left (x) word
        plaintext[:, 1] = right (y) word
    subkeys   : ndarray, shape (N, rounds) | (1, rounds) | (rounds,)
        Output of expand_key().
    n         : word width in bits.
    rounds    : round count T.

    Returns
    -------
    ndarray, shape (N, 2), same dtype.
    """
    x = plaintext[:, 0].copy()
    y = plaintext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]   # (1, rounds)

    for i in range(rounds):
        k       = subkeys[:, i]            # (N,) or (1,) — broadcasts
        x, y    = _enc_round(x, y, k, n)

    out      = np.empty_like(plaintext)
    out[:, 0] = x
    out[:, 1] = y
    return out


def encrypt_blocks_trace(
    plaintext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
) -> np.ndarray:
    """Encrypt and record (x, y) after each round, including the initial state.

    Returns
    -------
    ndarray, shape (rounds + 1, N, 2)
        trace[r, :, :] is the state after r encryption rounds (r=0 is plaintext).
    """
    x = plaintext[:, 0].copy()
    y = plaintext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]

    trace = np.empty((rounds + 1, plaintext.shape[0], 2), dtype=plaintext.dtype)
    trace[0, :, 0] = x
    trace[0, :, 1] = y

    for i in range(rounds):
        k    = subkeys[:, i]
        x, y = _enc_round(x, y, k, n)
        trace[i + 1, :, 0] = x
        trace[i + 1, :, 1] = y

    return trace


def decrypt_blocks(
    ciphertext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
) -> np.ndarray:
    """Decrypt N ciphertext blocks using pre-expanded subkeys.

    Parameters
    ----------
    ciphertext : ndarray, shape (N, 2), dtype compatible with n.
    subkeys    : ndarray, shape (N, rounds) | (1, rounds) | (rounds,).
    n          : word width in bits.
    rounds     : round count T.

    Returns
    -------
    ndarray, shape (N, 2), same dtype.
    """
    x = ciphertext[:, 0].copy()
    y = ciphertext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]

    for i in range(rounds - 1, -1, -1):
        k    = subkeys[:, i]
        x, y = _dec_round(x, y, k, n)

    out      = np.empty_like(ciphertext)
    out[:, 0] = x
    out[:, 1] = y
    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Simon:
    """SIMON block cipher — ECB raw-block encrypt/decrypt.

    ⚠️  For research, benchmarking, and education only.
        Not suitable for production cryptographic use.

    Parameters
    ----------
    n       : int
        Half-block / word width in bits.  Supported pairs listed in _SIMON_PARAMS.
    m       : int
        Number of key words.  Key size = n × m bits.
    z_index : int, optional
        Override z-sequence index (0–4).  Defaults to the spec value for (n, m).
    rounds  : int, optional
        Override number of rounds.  Defaults to the spec value for (n, m).

    Examples
    --------
    >>> import numpy as np
    >>> from simon import Simon
    >>> cipher = Simon(n=16, m=4)              # SIMON 32/64
    >>> key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    >>> pt  = np.array([[0x6565, 0x6877]], dtype=np.uint16)
    >>> ct  = cipher.encrypt(pt, key)
    >>> assert ct[0, 0] == 0xc69b and ct[0, 1] == 0xe9bb
    >>> assert (cipher.decrypt(ct, key) == pt).all()
    """

    def __init__(
        self,
        n: int = 16,
        m: int = 4,
        *,
        z_index: Optional[int] = None,
        rounds: Optional[int] = None,
    ) -> None:
        if (n, m) not in _SIMON_PARAMS:
            raise ValueError(
                f"(n={n}, m={m}) is not a valid SIMON parameter pair. "
                f"Supported pairs: {sorted(_SIMON_PARAMS)}"
            )
        spec_z, spec_rounds = _SIMON_PARAMS[(n, m)]
        self.n               = n
        self.m               = m
        self.z_index: int    = z_index if z_index is not None else spec_z
        self.rounds:  int    = rounds  if rounds  is not None else spec_rounds
        self.block_size_bits = 2 * n
        self.key_size_bits   = m * n
        self.dtype           = _dtype_for_n(n)

        if not (0 <= self.z_index <= 4):
            raise ValueError(f"z_index must be in 0–4, got {self.z_index}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

    # ------------------------------------------------------------------

    def _coerce_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 1 and data.shape[0] == 2:
            data = data[np.newaxis, :]
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                f"plaintext/ciphertext must have shape (N, 2) or (2,), got {data.shape}"
            )
        return data

    def _coerce_key(self, key: np.ndarray, N: int) -> np.ndarray:
        key = np.asarray(key, dtype=self.dtype)
        if key.ndim == 1:
            if key.shape[0] != self.m:
                raise ValueError(
                    f"1-D key must have {self.m} words, got {key.shape[0]}"
                )
            key = key[np.newaxis, :]
        if key.ndim != 2 or key.shape[-1] != self.m:
            raise ValueError(
                f"key must have shape ({self.m},), (1, {self.m}), or (N, {self.m}), "
                f"got {key.shape}"
            )
        if key.shape[0] not in (1, N):
            raise ValueError(
                f"key batch size {key.shape[0]} must be 1 or match plaintext N={N}"
            )
        return expand_key(key, self.n, self.m, self.rounds, self.z_index)

    # ------------------------------------------------------------------

    def encrypt(self, plaintext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Encrypt one or more blocks.

        Parameters
        ----------
        plaintext : array-like, shape (N, 2) or (2,)
            Column 0 = left (x) word; column 1 = right (y) word.
        key : array-like, shape (m,) | (1, m) | (N, m)
            Key words, big-endian word order: [k_{m-1}, …, k_0].

        Returns
        -------
        ndarray, shape (N, 2), dtype = self.dtype
        """
        pt = self._coerce_data(plaintext)
        sk = self._coerce_key(key, pt.shape[0])
        return encrypt_blocks(pt, sk, self.n, self.rounds)

    def decrypt(self, ciphertext: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Decrypt one or more blocks.

        Parameters
        ----------
        ciphertext : array-like, shape (N, 2) or (2,)
        key        : array-like, shape (m,) | (1, m) | (N, m)

        Returns
        -------
        ndarray, shape (N, 2), dtype = self.dtype
        """
        ct = self._coerce_data(ciphertext)
        sk = self._coerce_key(key, ct.shape[0])
        return decrypt_blocks(ct, sk, self.n, self.rounds)

    def __repr__(self) -> str:
        return (
            f"Simon(n={self.n}, m={self.m}, z_index={self.z_index}, "
            f"rounds={self.rounds}, block={self.block_size_bits}b, "
            f"key={self.key_size_bits}b)"
        )
