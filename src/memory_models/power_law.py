"""
Power-law memory model.

Equation:

H(t)=A(t+epsilon)^(-alpha)

Represents:

- anomalous transport
- diffusion-controlled memory
- heterogeneous porous media
"""

import numpy as np




def power_law_model(
        time,
        A,
        alpha,
        epsilon=1e-6
):
    """
    Calculate power-law memory.


    Parameters
    ----------

    time :
        time array


    A :
        amplitude


    alpha :
        power exponent


    epsilon :
        avoids singularity at zero



    Returns
    -------

    H(t)

    """

    time=np.asarray(
        time,
        dtype=float
    )


    if alpha <=0:

        raise ValueError(
            "alpha must be positive."
        )


    H=(

        A *

        (
            time+epsilon
        )**(-alpha)

    )


    return H
