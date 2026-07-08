"""
Hybrid memory model.

Equation:

H(t)=
A1 exp(-beta t)
+
A2(t+epsilon)^(-alpha)


Represents combined:

- fast exchange
- long-time anomalous memory

"""

import numpy as np





def hybrid_model(
        time,
        A1,
        beta,
        A2,
        alpha,
        epsilon=1e-6
):
    """
    Calculate hybrid memory.


    Parameters
    ----------

    time :
        time array


    A1 :
        exponential amplitude


    beta :
        exponential decay


    A2 :
        power-law amplitude


    alpha :
        power exponent



    Returns
    -------

    H(t)


    """

    time=np.asarray(
        time,
        dtype=float
    )


    if beta<=0:

        raise ValueError(
            "beta must be positive."
        )


    if alpha<=0:

        raise ValueError(
            "alpha must be positive."
        )


    exponential=(

        A1 *

        np.exp(
            -beta*time
        )

    )


    power=(

        A2 *

        (
            time+epsilon
        )**(-alpha)

    )


    return exponential+power
