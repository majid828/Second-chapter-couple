"""
Tests for memory kernel models.
"""


import numpy as np


from src.memory_models.exponential import (
    exponential_memory
)


from src.memory_models.power_law import (
    power_law_memory
)


from src.memory_models.hybrid_memory import (
    hybrid_memory
)



def test_exponential_memory():

    t=np.linspace(
        0,
        10,
        100
    )


    H=exponential_memory(

        t,

        amplitude=1,

        decay=0.2

    )


    assert len(H)==len(t)

    assert np.all(
        H>=0
    )



def test_power_law_memory():

    t=np.linspace(
        1,
        10,
        100
    )


    H=power_law_memory(

        t,

        amplitude=1,

        alpha=0.5

    )


    assert len(H)==100

    assert np.all(
        H>0
    )



def test_hybrid_memory():

    t=np.linspace(
        0,
        10,
        100
    )


    H=hybrid_memory(

        t,

        A1=0.5,

        beta=0.1,

        A2=0.5,

        alpha=0.5

    )


    assert len(H)==100

    assert np.all(
        H>=0
    )
