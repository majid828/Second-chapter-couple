"""
Noise sensitivity analysis.

Evaluates how measurement uncertainty
affects recovered memory functions.

"""

import numpy as np





def add_relative_noise(
        data,
        noise_level=0.05,
        random_seed=None
):
    """
    Add Gaussian relative noise.


    Parameters
    ----------

    data :
        original signal


    noise_level :
        fraction of maximum value


    Returns
    -------

    noisy data


    """

    if random_seed is not None:

        np.random.seed(
            random_seed
        )


    data=np.asarray(
        data,
        dtype=float
    )


    sigma = (
        noise_level *
        np.max(
            np.abs(data)
        )
    )


    noise=np.random.normal(
        0,
        sigma,
        size=data.shape
    )


    noisy=data+noise


    return np.maximum(
        noisy,
        0
    )





def noise_sensitivity_test(
        true_memory,
        recovered_memory
):
    """
    Calculate memory recovery error.

    Metric:

    NRMSE =
    ||H-Hhat|| /
    ||H||


    """

    true_memory=np.asarray(
        true_memory
    )

    recovered_memory=np.asarray(
        recovered_memory
    )


    numerator=np.linalg.norm(
        true_memory -
        recovered_memory
    )


    denominator=np.linalg.norm(
        true_memory
    )


    if denominator==0:

        raise ValueError(
            "True memory cannot be zero."
        )


    return numerator/denominator
