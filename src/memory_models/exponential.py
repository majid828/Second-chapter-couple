"""
Exponential memory model.

Equation:

H(t)=A exp(-beta t)

Represents:

- mobile-immobile exchange
- first order mass transfer
"""

import numpy as np



def exponential_model(
        time,
        A,
        beta
):
    """
    Calculate exponential memory.


    Parameters
    ----------
    time :
        time array


    A :
        amplitude


    beta :
        decay coefficient



    Returns
    -------

    H(t)

    """

    time=np.asarray(
        time,
        dtype=float
    )


    if beta <=0:

        raise ValueError(
            "beta must be positive."
        )


    H = (

        A *

        np.exp(
            -beta*time
        )

    )


    return H
