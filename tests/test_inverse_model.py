"""
Tests for inverse model constraints and regularization.
"""

import numpy as np

from src.inverse_model.constraints import (
    positivity_constraint,
    normalization_constraint,
    normalize_memory
)

from src.inverse_model.regularization import (
    first_order_difference_matrix
)


def test_positive_constraint():

    H = np.array(
        [0.1, 0.2, 0.3]
    )

    result = positivity_constraint(H)

    assert result is True



def test_normalization_constraint():

    time = np.linspace(
        0,
        10,
        100
    )

    H = np.exp(-0.3*time)

    H = normalize_memory(
        time,
        H
    )

    result = normalization_constraint(
        time,
        H
    )

    assert result is True



def test_regularization_matrix():

    L = first_order_difference_matrix(
        10
    )

    assert L.shape == (9,10)
