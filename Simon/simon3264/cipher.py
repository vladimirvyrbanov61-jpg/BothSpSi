"""SIMON 32/64 profile: frozen parameters, subkey cache, partial encryption."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from Simon.simon import Simon, decrypt_blocks, encrypt_blocks, expand_key

N_BITS = 16
M_WORDS = 4
BLOCK_BITS = 32
KEY_BITS = 64
ROUNDS = 32
Z_INDEX = 0


class Simon3264:
    """Frozen SIMON 32/64 profile (n=16, m=4, z0, T=32)."""

    n = N_BITS
    m = M_WORDS
    block_size_bits = BLOCK_BITS
    key_size_bits = KEY_BITS
    rounds = ROUNDS
    z_index = Z_INDEX
    dtype = np.dtype(np.uint16)

    def __init__(self) -> None:
        self._core = Simon(n=N_BITS, m=M_WORDS, z_index=Z_INDEX, rounds=ROUNDS)
        self._subkey_cache: dict[tuple[bytes, tuple[int, ...], int, int], np.ndarray] = {}

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
        z_index: Optional[int] = None,
    ) -> np.ndarray:
        """Expand key to round subkeys, shape (1, T) or (1, rounds) for variants."""
        r = ROUNDS if rounds is None else rounds
        z = Z_INDEX if z_index is None else z_index
        if not (1 <= r <= ROUNDS):
            raise ValueError(f"rounds must be in 1..{ROUNDS}, got {r}")
        if not 0 <= z <= 4:
            raise ValueError(f"z_index must be in 0..4, got {z}")
        k = self._coerce_key(key)
        key_bytes, key_shape = self._key_identity(k)
        cache_key = (key_bytes, key_shape, r, z)
        if use_cache and cache_key in self._subkey_cache:
            return self._subkey_cache[cache_key]
        sk = expand_key(k, N_BITS, M_WORDS, r, z)
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
        """Encrypt with *rounds* partial rounds (default: full T=32)."""
        r = ROUNDS if rounds is None else int(rounds)
        return self.encrypt_rounds(plaintext, key, r)

    def decrypt(
        self,
        ciphertext: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        ct = self._coerce_blocks(ciphertext)
        sk = self.get_subkeys(self._coerce_key(key, batch_size=len(ct)))
        return decrypt_blocks(ct, sk, N_BITS, ROUNDS)

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
        return encrypt_blocks(pt, subkeys, N_BITS, r)

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
        return encrypt_blocks(pt, sk, N_BITS, num_rounds)

    def encrypt_variant(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        rounds: Optional[int] = None,
        z_index: Optional[int] = None,
    ) -> np.ndarray:
        """Encrypt with non-default parameters (for synthetic fault injection)."""
        r = rounds if rounds is not None else ROUNDS
        z = z_index if z_index is not None else Z_INDEX
        if not (1 <= r <= ROUNDS):
            raise ValueError(f"rounds must be in 1..{ROUNDS}, got {r}")
        if not 0 <= z <= 4:
            raise ValueError(f"z_index must be in 0..4, got {z}")
        pt = self._coerce_blocks(plaintext)
        sk = self.get_subkeys(
            self._coerce_key(key, batch_size=len(pt)),
            use_cache=False,
            rounds=r,
            z_index=z,
        )
        if sk.shape[-1] > r:
            sk = sk[..., :r]
        return encrypt_blocks(pt, sk, N_BITS, r)

    def encrypt_ecb(
        self,
        plaintexts: np.ndarray,
        key: np.ndarray,
    ) -> np.ndarray:
        """ECB over independent blocks — same as encrypt."""
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
            f"Simon3264(block={BLOCK_BITS}b, key={KEY_BITS}b, "
            f"rounds={ROUNDS}, z_index={Z_INDEX})"
        )
