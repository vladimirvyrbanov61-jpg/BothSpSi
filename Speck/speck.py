"""
speck.py — SPECK block cipher (ECB-mode raw block functions only).

⚠️  PURPOSE: Research / benchmarking / education only.
    NOT for production cryptographic use.
    No constant-time guarantees; no side-channel mitigations.

Default profile: Speck(n=16, m=4) → SPECK 32/64, 22 rounds, α=7, β=2.

References
----------
Beaulieu et al., "The SIMON and SPECK Families of Lightweight Block Ciphers",
NSA / IACR Technical Report, 2013.  https://eprint.iacr.org/2013/404.pdf
NSA Cyber Implementation Guide v1.1 (little-endian byte/word order).

Public API
----------
Speck(n, m, *, rounds=None)
    .encrypt(plaintext, key) → ciphertext
    .decrypt(ciphertext, key) → plaintext

Array shapes / word ordering (aligned with simon.py)
----------------------------------------------------
plaintext / ciphertext : ndarray, shape (N, 2)
    axis 1 : [left_word, right_word] = [x, y]

key : ndarray, shape (N, m) | (1, m) | (m,)
    m words in big-endian key-word order:
        key[..., 0] = l_{m-2}  (most-significant key word in the schedule)
        ...
        key[..., m-1] = k_0   (least-significant / first round key)
    Internally reversed to [k_0, l_0, …, l_{m-2}] before expansion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# (n, m) → (alpha, beta, default_rounds)
_SPECK_PARAMS: dict[tuple[int, int], tuple[int, int, int]] = {
    (16, 4): (7, 2, 22),   # SPECK 32/64
    (24, 3): (8, 3, 22),   # SPECK 48/72
    (24, 4): (8, 3, 23),   # SPECK 48/96
    (32, 3): (8, 3, 26),   # SPECK 64/96
    (32, 4): (8, 3, 27),   # SPECK 64/128
    (48, 2): (8, 3, 28),   # SPECK 96/96
    (48, 3): (8, 3, 29),   # SPECK 96/144
    (64, 2): (8, 3, 32),   # SPECK 128/128
    (64, 3): (8, 3, 33),   # SPECK 128/192
    (64, 4): (8, 3, 34),   # SPECK 128/256
}


def _dtype_for_n(n: int) -> np.dtype:
    for bits, dt in [(8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)]:
        if n <= bits:
            return np.dtype(dt)
    raise ValueError(f"n={n} exceeds 64-bit word width")


def rol(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """Left-rotate each element of *x* by *k* bits within an n-bit word."""
    k = k % n
    if k == 0:
        return x.copy()
    dt = x.dtype
    mask = dt.type((1 << n) - 1)
    return ((x << dt.type(k)) | (x >> dt.type(n - k))) & mask


def ror(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """Right-rotate each element of *x* by *k* bits within an n-bit word."""
    return rol(x, n - (k % n), n)


def expand_key(
    key_words: np.ndarray,
    n: int,
    m: int,
    rounds: int,
    alpha: int,
    beta: int,
) -> np.ndarray:
    """Expand master key to *rounds* round keys k_0 … k_{rounds-1}.

    Parameters
    ----------
    key_words : ndarray, shape (..., m)
        Big-endian schedule order [l_{m-2}, …, l_0, k_0].
    n, m      : word width and number of key words.
    rounds    : round count T.
    alpha, beta : rotation constants for this (n, m) variant.

    Returns
    -------
    ndarray, shape (..., rounds)
        K[..., i] is the subkey for encryption round i.
    """
    dtype = _dtype_for_n(n)
    mask = dtype.type((1 << n) - 1)
    i_dtype = dtype.type(0)

    # Ascending schedule order: k_0, l_0, …, l_{m-2}
    key_asc = key_words[..., ::-1].astype(dtype)

    batch_shape = key_asc.shape[:-1]
    k_list: list[np.ndarray] = [key_asc[..., 0]]
    l_list: list[np.ndarray] = [key_asc[..., i] for i in range(1, m)]

    for i in range(rounds - 1):
        li = l_list[i]
        ki = k_list[i]
        round_const = i_dtype + dtype.type(i)
        l_new = (ki + ror(li, alpha, n)) ^ round_const
        l_new = l_new & mask
        k_new = rol(ki, beta, n) ^ l_new
        k_new = k_new & mask
        l_list.append(l_new)
        k_list.append(k_new)

    return np.stack(k_list, axis=-1)


def _enc_round(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    n: int,
    alpha: int,
    beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One SPECK encryption round (ARX, see Beaulieu et al. Fig. 1)."""
    mask = x.dtype.type((1 << n) - 1)
    x = ror(x, alpha, n)
    x = (x + y) & mask
    x = x ^ k
    y = rol(y, beta, n)
    y = y ^ x
    return x, y


def _dec_round(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    n: int,
    alpha: int,
    beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of one SPECK encryption round."""
    mask = x.dtype.type((1 << n) - 1)
    y_prev = ror(y ^ x, beta, n)
    t = (x ^ k).astype(x.dtype)
    # modular subtraction: (x' ⊕ k) − y  (mod 2^n)
    x_prev = rol((t - y_prev) & mask, alpha, n)
    return x_prev, y_prev


def encrypt_blocks(
    plaintext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
    alpha: int,
    beta: int,
) -> np.ndarray:
    """Encrypt N blocks with pre-expanded subkeys."""
    x = plaintext[:, 0].copy()
    y = plaintext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]

    for i in range(rounds):
        k = subkeys[:, i]
        x, y = _enc_round(x, y, k, n, alpha, beta)

    out = np.empty_like(plaintext)
    out[:, 0] = x
    out[:, 1] = y
    return out


def encrypt_blocks_trace(
    plaintext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
    alpha: int,
    beta: int,
) -> np.ndarray:
    """Encrypt and record state after each round; shape (rounds + 1, N, 2)."""
    x = plaintext[:, 0].copy()
    y = plaintext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]

    trace = np.empty((rounds + 1, plaintext.shape[0], 2), dtype=plaintext.dtype)
    trace[0, :, 0] = x
    trace[0, :, 1] = y

    for i in range(rounds):
        k = subkeys[:, i]
        x, y = _enc_round(x, y, k, n, alpha, beta)
        trace[i + 1, :, 0] = x
        trace[i + 1, :, 1] = y

    return trace


def decrypt_blocks(
    ciphertext: np.ndarray,
    subkeys: np.ndarray,
    n: int,
    rounds: int,
    alpha: int,
    beta: int,
) -> np.ndarray:
    """Decrypt N blocks with pre-expanded subkeys."""
    x = ciphertext[:, 0].copy()
    y = ciphertext[:, 1].copy()

    if subkeys.ndim == 1:
        subkeys = subkeys[np.newaxis, :]

    for i in range(rounds - 1, -1, -1):
        k = subkeys[:, i]
        x, y = _dec_round(x, y, k, n, alpha, beta)

    out = np.empty_like(ciphertext)
    out[:, 0] = x
    out[:, 1] = y
    return out


class Speck:
    """SPECK block cipher — ECB raw-block encrypt/decrypt (research only)."""

    def __init__(
        self,
        n: int = 16,
        m: int = 4,
        *,
        rounds: Optional[int] = None,
        alpha: Optional[int] = None,
        beta: Optional[int] = None,
    ) -> None:
        if (n, m) not in _SPECK_PARAMS:
            raise ValueError(
                f"unsupported (n, m)=({n}, {m}); supported: {sorted(_SPECK_PARAMS)}"
            )
        a, b, default_rounds = _SPECK_PARAMS[(n, m)]
        self.n = n
        self.m = m
        self.alpha = a if alpha is None else alpha
        self.beta = b if beta is None else beta
        self.rounds = default_rounds if rounds is None else rounds
        self.dtype = _dtype_for_n(n)
        self.block_size_bits = 2 * n
        self.key_size_bits = n * m
        if not isinstance(self.rounds, int) or self.rounds < 1:
            raise ValueError(f"rounds must be a positive integer, got {self.rounds}")
        for name, rotation in (("alpha", self.alpha), ("beta", self.beta)):
            if not isinstance(rotation, int) or not 1 <= rotation < n:
                raise ValueError(
                    f"{name} must be an integer in 1..{n - 1}, got {rotation}"
                )

    def _expand(
        self,
        key: np.ndarray,
        rounds: Optional[int] = None,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r = self.rounds if rounds is None else rounds
        k = np.asarray(key, dtype=self.dtype)
        if k.ndim == 1:
            if k.shape[0] != self.m:
                raise ValueError(f"key must have {self.m} words, got shape {k.shape}")
            k = k[np.newaxis, :]
        if k.ndim != 2 or k.shape[1] != self.m:
            raise ValueError(f"key must have {self.m} words, got shape {k.shape}")
        if batch_size is not None and k.shape[0] not in (1, batch_size):
            raise ValueError(
                f"key batch size {k.shape[0]} must be 1 or match block batch {batch_size}"
            )
        return expand_key(k, self.n, self.m, r, self.alpha, self.beta)

    def encrypt(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        pt = self._coerce_blocks(plaintext)
        sk = self._expand(key, batch_size=len(pt))
        return encrypt_blocks(pt, sk, self.n, self.rounds, self.alpha, self.beta)

    def decrypt(
        self,
        ciphertext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        ct = self._coerce_blocks(ciphertext)
        sk = self._expand(key, batch_size=len(ct))
        return decrypt_blocks(ct, sk, self.n, self.rounds, self.alpha, self.beta)

    def _coerce_blocks(self, data: np.ndarray) -> np.ndarray:
        d = np.asarray(data, dtype=self.dtype)
        if d.ndim == 1 and d.shape[0] == 2:
            d = d[np.newaxis, :]
        if d.ndim != 2 or d.shape[1] != 2:
            raise ValueError(f"expected shape (N, 2) or (2,), got {d.shape}")
        return d

    def __repr__(self) -> str:
        return (
            f"Speck(n={self.n}, m={self.m}, rounds={self.rounds}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )
