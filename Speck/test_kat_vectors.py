"""Official SPECK KAT word values (paper Appendix C / Crypto++ / Implementation Guide)."""

from __future__ import annotations

import numpy as np


def _word_from_hex(hex_str: str, n: int) -> int:
    if n in (32, 64):
        return int.from_bytes(bytes.fromhex(hex_str), "little")
    return int(hex_str, 16)


def _words(hex_words: list[str], n: int) -> list[int]:
    return [_word_from_hex(h, n) for h in hex_words]


_RAW: list[tuple[int, int, list[str], list[str], list[str]]] = [
    (16, 4, ["6574", "694C"], ["a868", "42f2"], ["1918", "1110", "0908", "0100"]),
    (24, 3, ["20796c", "6c6172"], ["c049a5", "385adc"], ["121110", "0a0908", "020100"]),
    (
        24,
        4,
        ["6d2073", "696874"],
        ["735e10", "b6445d"],
        ["1a1918", "121110", "0a0908", "020100"],
    ),
    (32, 3, ["65616E73", "20466174"], ["6C947541", "EC52799F"], ["00010203", "08090A0B", "10111213"]),
    (32, 4, ["2D437574", "7465723B"], ["8B024E45", "48A56F8C"], ["00010203", "08090A0B", "10111213", "18191A1B"]),
    (
        48,
        2,
        ["65776f68202c", "656761737520"],
        ["9e4d09ab7178", "62bdde8f79aa"],
        ["0d0c0b0a0908", "050403020100"],
    ),
    (
        48,
        3,
        ["656d6974206e", "69202c726576"],
        ["2bf31072228a", "7ae440252ee6"],
        ["151413121110", "0d0c0b0a0908", "050403020100"],
    ),
    (
        64,
        2,
        ["206D616465206974", "206571756976616C"],
        ["180D575CDFFE6078", "6532787951985DA6"],
        ["0001020304050607", "08090A0B0C0D0E0F"],
    ),
    (
        64,
        3,
        ["656E7420746F2043", "6869656620486172"],
        ["86183CE05D18BCF9", "665513133ACFE41B"],
        ["0001020304050607", "08090A0B0C0D0E0F", "1011121314151617"],
    ),
    (
        64,
        4,
        ["706F6F6E65722E20", "496E2074686F7365"],
        ["438F189C8DB4EE4E", "3EF5C00504010941"],
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
    pt = _words(pt_h, n)
    ct = _words(ct_h, n)
    key = _words(key_h, n)
    if n in (32, 64):
        pt = [pt[1], pt[0]]
        ct = [ct[1], ct[0]]
        key = list(reversed(key))
    return pt, ct, key


SPECK_OFFICIAL_KATS: list[tuple[int, int, list[int], list[int], list[int]]] = [
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
