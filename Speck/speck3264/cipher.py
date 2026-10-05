"""SPECK 32/64 profile: frozen parameters, subkey cache, partial encryption."""

from __future__ import annotations

from typing import Optional

import numpy as np

from Speck.speck import Speck, decrypt_blocks, encrypt_blocks, expand_key

N_BITS = 16
M_WORDS = 4
BLOCK_BITS = 32
KEY_BITS = 64
ROUNDS = 22
ALPHA = 7
BETA = 2


class Speck3264:
    """Frozen SPECK 32/64 profile (n=16, m=4, α=7, β=2, T=22)."""

    n = N_BITS
    m = M_WORDS
    block_size_bits = BLOCK_BITS
    key_size_bits = KEY_BITS
    rounds = ROUNDS
    alpha = ALPHA
    beta = BETA
    dtype = np.dtype(np.uint16)

    def __init__(self) -> None:
        self._core = Speck(n=N_BITS, m=M_WORDS, rounds=ROUNDS)
        self._subkey_cache: dict[tuple[bytes, tuple[int, ...], int], np.ndarray] = {}

    def _key_identity(self, key: np.ndarray) -> tuple[bytes, tuple[int, ...]]:
        k = np.ascontiguousarray(np.asarray(key, dtype=self.dtype))
        return k.tobytes(), k.shape

    def _coerce_key(
        self,
        key: np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        k = np.asarray(key, dtype=self.dtype)
        if k.ndim == 1:
            if k.shape[0] != M_WORDS:
                raise ValueError(f"key must contain {M_WORDS} words, got {k.shape}")
            k = k[np.newaxis, :]
        if k.ndim != 2 or k.shape[1] != M_WORDS:
            raise ValueError(
                f"key must have shape ({M_WORDS},), (1, {M_WORDS}), "
                f"or (N, {M_WORDS})"
            )
        if batch_size is not None and k.shape[0] not in (1, batch_size):
            raise ValueError(
                f"key batch size {k.shape[0]} must be 1 or match block batch {batch_size}"
            )
        return k

    def get_subkeys(
        self,
        key: np.ndarray,
        *,
        use_cache: bool = True,
        rounds: Optional[int] = None,
    ) -> np.ndarray:
        """Expand key to round subkeys, shape (1, T) or (1, rounds)."""
        r = ROUNDS if rounds is None else rounds
        if not (1 <= r <= ROUNDS):
            raise ValueError(f"rounds must be in 1..{ROUNDS}, got {r}")
        k = self._coerce_key(key)
        key_bytes, key_shape = self._key_identity(k)
        cache_key = (key_bytes, key_shape, r)
        if use_cache and cache_key in self._subkey_cache:
            return self._subkey_cache[cache_key]
        sk = expand_key(k, N_BITS, M_WORDS, r, ALPHA, BETA)
        if use_cache:
            self._subkey_cache[cache_key] = sk
        return sk

    def clear_subkey_cache(self) -> None:
        self._subkey_cache.clear()

    def encrypt(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        rounds: Optional[int] = None,
    ) -> np.ndarray:
        """Encrypt with *rounds* partial rounds (default: full T=22)."""
        r = ROUNDS if rounds is None else int(rounds)
        return self.encrypt_rounds(plaintext, key, r)

    def decrypt(
        self,
        ciphertext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        ct = self._coerce_blocks(ciphertext)
        sk = self.get_subkeys(self._coerce_key(key, batch_size=len(ct)))
        return decrypt_blocks(ct, sk, N_BITS, ROUNDS, ALPHA, BETA)

    def encrypt_with_subkeys(
        self,
        plaintext: np.ndarray,
        subkeys: np.ndarray,
    ) -> np.ndarray:
        pt = self._coerce_blocks(plaintext)
        if subkeys.ndim not in (1, 2) or not (1 <= subkeys.shape[-1] <= ROUNDS):
            raise ValueError(f"subkeys must contain between 1 and {ROUNDS} rounds")
        if subkeys.ndim == 2 and subkeys.shape[0] not in (1, len(pt)):
            raise ValueError("subkey batch must be 1 or match the block batch")
        r = subkeys.shape[-1]
        return encrypt_blocks(pt, subkeys, N_BITS, r, ALPHA, BETA)

    def encrypt_rounds(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        num_rounds: int,
    ) -> np.ndarray:
        if not (1 <= num_rounds <= ROUNDS):
            raise ValueError(f"num_rounds must be in 1..{ROUNDS}, got {num_rounds}")
        pt = self._coerce_blocks(plaintext)
        sk = self.get_subkeys(
            self._coerce_key(key, batch_size=len(pt)),
            rounds=num_rounds,
        )[:, :num_rounds]
        return encrypt_blocks(pt, sk, N_BITS, num_rounds, ALPHA, BETA)

    def encrypt_ecb(
        self,
        plaintexts: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        return self.encrypt(plaintexts, key)

    def _coerce_blocks(self, data: np.ndarray) -> np.ndarray:
        d = np.asarray(data, dtype=self.dtype)
        if d.ndim == 1 and d.shape[0] == 2:
            d = d[np.newaxis, :]
        if d.ndim != 2 or d.shape[1] != 2:
            raise ValueError(f"expected shape (N, 2) or (2,), got {d.shape}")
        return d

    def __repr__(self) -> str:
        return (
            f"Speck3264(block={BLOCK_BITS}b, key={KEY_BITS}b, "
            f"rounds={ROUNDS}, alpha={ALPHA}, beta={BETA})"
        )
