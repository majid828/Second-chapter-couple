"""
Tests for plume reconstruction.
"""


import numpy as np


from src.reconstruction.error_metrics import (
    rmse,
    mae
)



def test_rmse():

    observed=np.array(
        [1,2,3]
    )


    predicted=np.array(
        [1,2,4]
    )


    value=rmse(

        observed,

        predicted

    )


    assert np.isclose(

        value,

        np.sqrt(1/3)

    )



def test_mae():

    observed=np.array(
        [1,2,3]
    )


    predicted=np.array(
        [1,2,4]
    )


    value=mae(

        observed,

        predicted

    )


    assert np.isclose(

        value,

        1/3

    )
