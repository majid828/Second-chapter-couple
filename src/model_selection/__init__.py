"""
Model selection module.

Provides:

- Akaike Information Criterion
- Bayesian Information Criterion
- Cross-validation comparison


Used for selecting the most appropriate
transport memory mechanism.

"""


from .aic import (
    calculate_aic
)

from .bic import (
    calculate_bic
)

from .cross_validation import (
    cross_validate_memory_model
)


__all__=[

    "calculate_aic",

    "calculate_bic",

    "cross_validate_memory_model"

]
