"""
Validation module.

Contains:

1. Synthetic validation
2. Field-data validation

"""

from .synthetic_validation import (
    validate_memory_recovery
)

from .field_validation import (
    validate_field_prediction
)


__all__ = [

    "validate_memory_recovery",

    "validate_field_prediction"

]
