"""
Uncertainty quantification module.

Provides:

- Bayesian posterior covariance
- Bootstrap confidence intervals
- Noise sensitivity analysis

"""

from .posterior_covariance import (
    compute_posterior_covariance,
    memory_confidence_interval
)

from .bootstrap import (
    bootstrap_memory_recovery
)

from .noise_analysis import (
    add_relative_noise,
    noise_sensitivity_test
)


__all__ = [

    "compute_posterior_covariance",

    "memory_confidence_interval",

    "bootstrap_memory_recovery",

    "add_relative_noise",

    "noise_sensitivity_test"

]
