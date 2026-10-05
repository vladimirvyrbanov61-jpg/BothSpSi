"""Classical DDT baselines and non-exhaustive characteristic estimates."""

from thesis.classical.characteristic import (
    estimate_max_characteristic_probability,
    max_characteristic_probability,
    track_characteristic_over_rounds,
)
from thesis.classical.ddt_core import Delta32, delta32_to_key, key_to_delta32
from thesis.classical.ddt_simon import (
    compute_f_ddt_exact,
    compute_simon_round_ddt,
    simon_round_transition_monte_carlo,
)
from thesis.classical.ddt_speck import (
    compute_speck_round_ddt,
    speck_round_transition_monte_carlo,
)

__all__ = [
    "Delta32",
    "delta32_to_key",
    "key_to_delta32",
    "compute_f_ddt_exact",
    "compute_simon_round_ddt",
    "simon_round_transition_monte_carlo",
    "compute_speck_round_ddt",
    "speck_round_transition_monte_carlo",
    "estimate_max_characteristic_probability",
    "max_characteristic_probability",
    "track_characteristic_over_rounds",
]
