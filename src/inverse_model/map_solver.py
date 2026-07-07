"""
MAP solver for constrained memory recovery.

Solves:

min_H

0.5(f-KH)^T Sigma^-1(f-KH)

+

lambda/2 ||LH||^2


subject to:

H>=0

"""

import numpy as np

import cvxpy as cp





def solve_map_problem(
        K,
        f,
        L,
        lam=1e-3,
        noise_variance=1e-4
):
    """
    Solve constrained MAP inverse problem.


    Parameters
    ----------

    K :
        convolution matrix


    f :
        normalized BTC


    L :
        regularization matrix


    lam :
        regularization strength


    noise_variance :
        measurement variance



    Returns
    -------

    H_est

    """



    K=np.asarray(K)

    f=np.asarray(f)

    L=np.asarray(L)



    n=K.shape[1]


    # unknown memory vector

    H=cp.Variable(
        n
    )


    residual = (
        f-K@H
    )


    data_term = cp.sum_squares(
        residual
    )/(2*noise_variance)


    regularization = (
        lam/2 *
        cp.sum_squares(
            L@H
        )
    )


    objective = cp.Minimize(
        data_term+
        regularization
    )


    constraints=[

        H>=0

    ]


    problem=cp.Problem(
        objective,
        constraints
    )


    problem.solve(
        solver=cp.OSQP
    )


    if H.value is None:

        raise RuntimeError(
            "MAP optimization failed."
        )


    H_solution=np.array(
        H.value
    )


    # finite-window normalization

    integral=np.sum(
        H_solution
    )


    if integral>0:

        H_solution /= integral


    return H_solution
