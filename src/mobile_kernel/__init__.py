"""
Mobile transport kernel module.

This module implements:

1. Parametric mobile travel-time kernel
2. Convolution matrix construction
3. Parameter optimization

The mobile kernel is defined as:

g(t)=
t^m exp(-b t)
-----------------
integral(t^m exp(-b t)dt)

"""

from .parametric_kernel import (
    mobile_kernel,
    normalize_kernel
)

from .kernel_matrix import (
    build_convolution_matrix
)

from .optimize_mobile_kernel import (
    optimize_mobile_parameters
)


__all__ = [

    "mobile_kernel",
    "normalize_kernel",

    "build_convolution_matrix",

    "optimize_mobile_parameters"

]
