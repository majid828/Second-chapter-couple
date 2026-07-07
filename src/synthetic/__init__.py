"""
Synthetic transport experiment generator.

This module provides functions for:

- generating temporal memory functions
- generating spatial transport kernels
- creating synthetic concentration fields
- adding measurement noise

"""

from .generate_memory import (
    exponential_memory,
    power_law_memory,
    hybrid_memory
)

from .generate_transport import (
    gaussian_transport_kernel,
    generate_concentration_field
)

from .add_noise import (
    add_gaussian_noise
)


__all__ = [

    "exponential_memory",
    "power_law_memory",
    "hybrid_memory",

    "gaussian_transport_kernel",
    "generate_concentration_field",

    "add_gaussian_noise"

]
