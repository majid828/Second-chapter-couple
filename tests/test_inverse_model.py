"""
Tests for inverse memory recovery.
"""


import numpy as np


from src.inverse_model.constraints import (
    enforce_non_negative
)


from src.inverse_model.regularization import (
    first_order_difference_matrix
)



def test_non_negative_constraint():

    """

    Test positivity constraint.

    """


    x=np.array(
        [-1,0,2]
    )


    result=enforce_non_negative(x)


    assert np.all(
        result>=0
    )



def test_regularization_matrix():

    """

    Test regularization operator.

    """


    L=first_order_difference_matrix(
        10
    )


    assert L.shape==(9,10)
