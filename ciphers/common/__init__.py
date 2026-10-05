"""Bit encoding and uniform sampling for 32/64 block ciphers."""

from ciphers.common.encoding import (
    BLOCK_BITS,
    WORD_BITS,
    block_to_bits,
    blocks_to_bits,
    concat_pair_bits,
    concat_pairs_batch,
)
from ciphers.common.sampling import (
    apply_delta,
    random_blocks,
    sample_keys,
    sample_plaintexts,
)

__all__ = [
    "BLOCK_BITS",
    "WORD_BITS",
    "block_to_bits",
    "blocks_to_bits",
    "concat_pair_bits",
    "concat_pairs_batch",
    "apply_delta",
    "random_blocks",
    "sample_keys",
    "sample_plaintexts",
]
