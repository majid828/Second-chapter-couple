"""
Noise generator for synthetic experiments.

Used for testing robustness of
memory recovery under measurement uncertainty.

"""

import numpy as np





def add_gaussian_noise(
        data,
        noise_level=0.05,
        random_seed=None
):
    """
    Add Gaussian measurement noise.

    Parameters
    ----------

    data :
        original signal

    noise_level :
        relative noise amplitude

    random_seed :
        reproducibility


    Returns
    -------

    noisy data

    """

    if random_seed is not None:

        np.random.seed(
            random_seed
        )


    data = np.asarray(data)


    sigma = (
        noise_level *
        np.max(
            np.abs(data)
        )
    )


    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=data.shape
    )


    noisy = data + noise


    # concentration cannot be negative

    noisy = np.maximum(
        noisy,
        0
    )


    return noisy
