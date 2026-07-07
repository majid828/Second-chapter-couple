"""
Posterior uncertainty calculation for Bayesian memory recovery.

The posterior covariance is:

C_H =
(K^T Sigma^-1 K + lambda L^T L)^-1


The diagonal elements provide variance estimates
for recovered memory values.

"""

import numpy as np





def compute_posterior_covariance(
        K,
        L,
        noise_variance=1e-4,
        lam=1e-3
):
    """
    Compute posterior covariance matrix.

    Parameters
    ----------
    K :
        convolution matrix


    L :
        regularization matrix


    noise_variance :
        measurement variance


    lam :
        regularization parameter


    Returns
    -------
    covariance matrix


    """

    K = np.asarray(K)

    L = np.asarray(L)



    n = K.shape[1]


    # Sigma^-1
    Sigma_inv = (
        1.0 /
        noise_variance
    )


    precision_matrix = (

        Sigma_inv *
        (K.T @ K)

        +

        lam *
        (L.T @ L)

    )


    # numerical stabilization

    precision_matrix += (
        1e-12 *
        np.eye(n)
    )


    covariance = np.linalg.inv(
        precision_matrix
    )


    return covariance





def memory_confidence_interval(
        memory,
        covariance,
        confidence=0.95
):
    """
    Calculate confidence interval
    for recovered memory.


    Parameters
    ----------
    memory :
        recovered H(t)


    covariance :
        posterior covariance


    confidence :
        confidence level


    Returns
    -------

    lower bound

    upper bound


    """

    memory = np.asarray(
        memory
    )


    variance = np.diag(
        covariance
    )


    std = np.sqrt(
        np.maximum(
            variance,
            0
        )
    )


    # Gaussian confidence multiplier

    if confidence == 0.95:

        multiplier = 1.96

    elif confidence == 0.99:

        multiplier = 2.576

    else:

        raise ValueError(
            "Only 0.95 and 0.99 confidence supported."
        )


    lower = (
        memory -
        multiplier*std
    )


    upper = (
        memory +
        multiplier*std
    )


    return lower, upper
