"""Resolve Simon32/64 and Speck32/64 profile ciphers by name."""

from __future__ import annotations

from typing import Literal, Protocol, Union

import numpy as np

CipherName = Literal["simon", "speck"]


class BlockCipherProfile(Protocol):
    """Minimal interface for thesis data generation."""

    rounds: int
    dtype: np.dtype

    def encrypt(
        self,
        plaintext: np.ndarray,
        key: np.ndarray,
        *,
        rounds: int | None = None,
    ) -> np.ndarray: ...

    def clear_subkey_cache(self) -> None: ...


def get_cipher(name: CipherName) -> Union["Simon3264", "Speck3264"]:
    if name == "simon":
        from Simon.simon3264.cipher import Simon3264

        return Simon3264()
    if name == "speck":
        from Speck.speck3264.cipher import Speck3264

        return Speck3264()
    raise ValueError(f"unknown cipher {name!r}; use 'simon' or 'speck'")


def max_rounds(name: CipherName) -> int:
    return int(get_cipher(name).rounds)
