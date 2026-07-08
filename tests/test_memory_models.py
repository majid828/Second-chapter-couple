"""
Tests for memory kernel models.
"""


import numpy as np


from src.memory_models.exponential import (
    exponential_model
)

from src.memory_models.power_law import (
    power_law_model
)

from src.memory_models.hybrid_memory import (
    hybrid_model
)



def test_exponential_model():

    time = np.linspace(
        0,
        10,
        100
    )


    H = exponential_model(
        time,
        A=1.0,
        beta=0.2
    )


    assert len(H)==len(time)

    assert np.all(
        H >= 0
    )



def test_power_law_model():

    time=np.linspace(
        1,
        10,
        100
    )


    H = power_law_model(
        time,
        A=1.0,
        alpha=0.5
    )


    assert len(H)==100

    assert np.all(
        H > 0
    )



def test_hybrid_model():

    time=np.linspace(
        0,
        10,
        100
    )


    H = hybrid_model(
        time,
        A1=0.5,
        beta=0.1,
        A2=0.5,
        alpha=0.5
    )


    assert len(H)==100

    assert np.all(
        H >= 0
    )
