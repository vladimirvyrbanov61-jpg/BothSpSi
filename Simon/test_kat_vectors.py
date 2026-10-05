"""Official SIMON KAT word values (paper Appendix B / Crypto++ / Implementation Guide).

Hex strings are little-endian byte encodings per word (NSA Implementation Guide).
Two words are listed in wire order: first = left (x), second = right (y).
Keys are big-endian schedule order [k_{m-1}, …, k_0].
"""

from __future__ import annotations

import numpy as np


def _word_from_hex(hex_str: str, n: int) -> int:
    """Parse appendix / Crypto++ hex to an n-bit word.

    n in {16, 24, 48}: appendix lists the word value (``1918`` → 0x1918).
    n in {32, 64}: Implementation Guide / Crypto++ use little-endian byte encoding.
    """
    if n in (32, 64):
        return int.from_bytes(bytes.fromhex(hex_str), "little")
    # n in {16, 24, 48}: appendix word is the integer printed in hex (Crypto++).
    return int(hex_str, 16)


def _words(hex_words: list[str], n: int) -> list[int]:
    return [_word_from_hex(h, n) for h in hex_words]


# (n, m, plain_hex_pair, cipher_hex_pair, key_hex_words)
_RAW: list[tuple[int, int, list[str], list[str], list[str]]] = [
    # SIMON 32/64
    (16, 4, ["6565", "6877"], ["c69b", "e9bb"], ["1918", "1110", "0908", "0100"]),
    # SIMON 48/72 (3-byte words)
    (24, 3, ["612067", "6e696c"], ["dae5ac", "292cac"], ["121110", "0a0908", "020100"]),
    # SIMON 48/96
    (
        24,
        4,
        ["726963", "20646e"],
        ["6e06a5", "acf156"],
        ["1a1918", "121110", "0a0908", "020100"],
    ),
    # SIMON 64/96
    (32, 3, ["636C696E", "6720726F"], ["C88F1A11", "7FE2A25C"], ["00010203", "08090A0B", "10111213"]),
    # SIMON 64/128
    (32, 4, ["756E6420", "6C696B65"], ["7AA0DFB9", "20FCC844"], ["00010203", "08090A0B", "10111213", "18191A1B"]),
    # SIMON 96/96 (6-byte words)
    (
        48,
        2,
        ["2072616c6c69", "702065687420"],
        ["602807a462b4", "69063d8ff082"],
        ["0d0c0b0a0908", "050403020100"],
    ),
    # SIMON 96/144
    (
        48,
        3,
        ["746168742074", "73756420666f"],
        ["ecad1c6c451e", "3f59c5db1ae9"],
        ["151413121110", "0d0c0b0a0908", "050403020100"],
    ),
    # SIMON 128/128
    (
        64,
        2,
        ["2074726176656C6C", "6572732064657363"],
        ["BC0B4EF82A83AA65", "3FFE541E1E1B6849"],
        ["0001020304050607", "08090A0B0C0D0E0F"],
    ),
    # SIMON 128/192
    (
        64,
        3,
        ["7269626520776865", "6E20746865726520"],
        ["5BB897256E8D9C6C", "4F0DDCFCEF61ACC4"],
        ["0001020304050607", "08090A0B0C0D0E0F", "1011121314151617"],
    ),
    # SIMON 128/256
    (
        64,
        4,
        ["697320612073696D", "6F6F6D20696E2074"],
        ["68B8E7EF872AF73B", "A0A3C8AF79552B8D"],
        [
            "0001020304050607",
            "08090A0B0C0D0E0F",
            "1011121314151617",
            "18191A1B1C1D1E1F",
        ],
    ),
]

def _kat_words(
    pt_h: list[str], ct_h: list[str], key_h: list[str], n: int
) -> tuple[list[int], list[int], list[int]]:
    """Build [left, right] word arrays for this implementation.

    n in {16, 24, 48}: words are appendix integers (no swap).
    n in {32, 64}: each word is little-endian bytes (Crypto++); swap columns
    to [high-address word, low-address word] and reverse key to k_{m-1}…k_0.
    """
    pt = _words(pt_h, n)
    ct = _words(ct_h, n)
    key = _words(key_h, n)
    if n in (32, 64):
        pt = [pt[1], pt[0]]
        ct = [ct[1], ct[0]]
        key = list(reversed(key))
    return pt, ct, key


SIMON_OFFICIAL_KATS: list[tuple[int, int, list[int], list[int], list[int]]] = [
    (n, m, *_kat_words(pt_h, ct_h, key_h, n)) for n, m, pt_h, ct_h, key_h in _RAW
]


def dtype_for_n(n: int) -> np.dtype:
    for bits, dt in [(16, np.uint16), (32, np.uint32), (64, np.uint64)]:
        if n <= bits:
            return np.dtype(dt)
    raise ValueError(f"unsupported word width n={n}")


def kat_arrays(
    n: int,
    plain: list[int],
    cipher: list[int],
    key: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = dtype_for_n(n)
    return (
        np.array(plain, dtype=dt),
        np.array(cipher, dtype=dt),
        np.array(key, dtype=dt),
    )
