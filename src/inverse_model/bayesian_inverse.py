"""
High-level Bayesian inverse model.

Recover temporal memory kernel H(t)
from observed BTC:

f = K H

using constrained MAP estimation.
"""


import numpy as np


from .map_solver import (
    solve_map_problem
)


from .regularization import (
    second_order_regularization
)



def recover_memory_kernel(
        time,
        btc,
        convolution_matrix,
        lam=1e-5,
        noise_variance=None,
        regularization_order=2
):
    """
    Recover temporal memory kernel.

    Parameters
    ----------
    time :
        time vector

    btc :
        observed breakthrough curve

    convolution_matrix :
        transport convolution matrix

    lam :
        regularization parameter

    noise_variance :
        measurement variance


    Returns
    -------
    dict

        memory

        regularization_matrix

    """



    time = np.asarray(
        time,
        dtype=float
    )


    btc = np.asarray(
        btc,
        dtype=float
    )


    n = len(time)



    # ------------------------------
    # Regularization operator
    # ------------------------------

    if regularization_order == 2:

        L = second_order_regularization(
            n
        )

    else:

        raise ValueError(
            "Only second order regularization is implemented."
        )



    # ------------------------------
    # Estimate noise variance
    # ------------------------------

    if noise_variance is None:

        noise_variance = np.var(
            btc
        )



    # ------------------------------
    # MAP recovery
    # ------------------------------

    H = solve_map_problem(

        convolution_matrix,

        btc,

        L,

        time,

        lam,

        noise_variance

    )



    return {

        "time": time,

        "memory": H,

        "regularization_matrix": L,

        "lambda": lam,

        "noise_variance": noise_variance

    }
