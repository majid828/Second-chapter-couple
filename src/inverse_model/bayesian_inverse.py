"""
High-level Bayesian memory recovery interface.

This module combines:

- convolution matrix
- regularization
- MAP solver
- physical constraints


Input:

BTC f(t)

Mobile kernel matrix K


Output:

Recovered memory function H(t)

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
        lam=1e-3,
        noise_variance=1e-4,
        regularization_order=2
):
    """
    Recover temporal memory function.


    Parameters
    ----------

    time :
        observation time


    btc :
        normalized breakthrough curve


    convolution_matrix :
        mobile kernel convolution matrix



    Returns
    -------

    dictionary:

        memory

        regularization matrix


    """


    n=len(time)



    if regularization_order==1:

        from .regularization import (
            first_order_regularization
        )

        L=first_order_regularization(
            n
        )


    else:

        L=second_order_regularization(
            n
        )



    H=solve_map_problem(

        convolution_matrix,

        btc,

        L,

        lam,

        noise_variance

    )



    return {

        "time":time,

        "memory":H,

        "regularization_matrix":L,

        "lambda":lam,

        "noise_variance":noise_variance

    }
