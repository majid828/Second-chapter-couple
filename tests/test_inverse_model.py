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
    second_order_regularization
)


def test_positive_constraint():

    H = np.array(
        [0.1, 0.2, 0.3]
    )

    result = positivity_constraint(H)

    assert bool(result) is True



def test_normalization_constraint():

    time = np.linspace(
        0,
        10,
        200
    )

    # create positive memory function
    H = np.exp(-0.3*time)

    # normalize
    H = normalize_memory(
        time,
        H
    )

    result = normalization_constraint(
        time,
        H
    )

    assert bool(result) is True



def test_second_order_regularization():

    n = 10

    L = second_order_regularization(
        n
    )

    # second derivative matrix shape
    assert L.shape == (
        n-2,
        n
    )

    # check matrix construction
    assert L[0,0] == 1
    assert L[0,1] == -2
    assert L[0,2] == 1
