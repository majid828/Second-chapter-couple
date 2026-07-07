"""
Regularization operators for inverse memory recovery.

Tikhonov regularization:

||L H||^2

is used to stabilize the ill-conditioned
deconvolution problem.

"""

import numpy as np




def first_order_regularization(
        n
):
    """
    First derivative smoothing matrix.

    Penalizes abrupt changes:

    ||dH/dt||^2


    """

    L=np.zeros(
        (n-1,n)
    )


    for i in range(n-1):

        L[i,i]=-1

        L[i,i+1]=1


    return L





def second_order_regularization(
        n
):
    """
    Second derivative smoothing matrix.

    Penalizes curvature:

    ||d2H/dt2||^2


    """

    L=np.zeros(
        (n-2,n)
    )


    for i in range(n-2):

        L[i,i]=1

        L[i,i+1]=-2

        L[i,i+2]=1


    return L
