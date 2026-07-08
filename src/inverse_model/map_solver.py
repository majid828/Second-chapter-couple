"""
Physically constrained MAP solver for transport memory recovery.

Solves:

min_H

0.5 ||K H - f||^2_sigma
+
lambda/2 ||L H||^2


Subject to:

H >= 0

Integral(H)=1

"""

import numpy as np
import cvxpy as cp



def solve_map_problem(
        K,
        f,
        L,
        time,
        lam=1e-3,
        noise_variance=None
):

    K=np.asarray(K,dtype=float)

    f=np.asarray(f,dtype=float)

    L=np.asarray(L,dtype=float)

    time=np.asarray(time,dtype=float)


    n=K.shape[1]


    dt=np.mean(
        np.diff(time)
    )


    if noise_variance is None:

        noise_variance=np.var(
            f
        )


    H=cp.Variable(
        n,
        nonneg=True
    )


    residual=f-K@H


    data_term=(

        cp.sum_squares(residual)

        /

        (2*noise_variance)

    )


    reg_term=(

        lam/2

        *

        cp.sum_squares(
            L@H
        )

    )


    objective=cp.Minimize(

        data_term+reg_term

    )


    constraints=[

        cp.sum(H)*dt == 1

    ]


    problem=cp.Problem(

        objective,

        constraints

    )


    problem.solve(

        solver=cp.CLARABEL

    )


    if H.value is None:

        raise RuntimeError(

            "MAP solver failed"

        )


    H_est=np.asarray(
        H.value
    )


    return H_est
