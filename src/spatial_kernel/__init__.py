"""
Spatial transport kernel analysis module.

Functions:

- Convert concentration fields into spatial probability kernels
- Calculate characteristic functions
- Calculate plume statistics
- Quantify non-Gaussian transport behavior

"""

from .spatial_probability import (
    spatial_probability_density
)

from .characteristic_function import (
    characteristic_function,
    inverse_characteristic_statistics
)

from .plume_statistics import (
    plume_centroid,
    plume_variance,
    plume_moments
)

from .non_gaussian import (
    skewness,
    kurtosis,
    non_gaussian_metrics
)


__all__ = [

    "spatial_probability_density",

    "characteristic_function",

    "inverse_characteristic_statistics",

    "plume_centroid",

    "plume_variance",

    "plume_moments",

    "skewness",

    "kurtosis",

    "non_gaussian_metrics"

]
