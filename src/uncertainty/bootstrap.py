"""
Bootstrap uncertainty analysis.

The idea:

1. Perturb observed BTC.
2. Recover memory repeatedly.
3. Calculate distribution of H(t).

This evaluates stability of the inverse solution.

"""

import numpy as np





def bootstrap_memory_recovery(
        btc,
        recovery_function,
        n_bootstrap=100,
        noise_level=0.05,
        random_seed=None
):
    """
    Bootstrap memory recovery.


    Parameters
    ----------

    btc :
        observed normalized BTC


    recovery_function :
        function that accepts BTC
        and returns memory vector


    n_bootstrap :
        number of realizations


    noise_level :
        relative perturbation


    Returns
    -------

    bootstrap_memory_samples


    """

    if random_seed is not None:

        np.random.seed(
            random_seed
        )


    btc = np.asarray(
        btc
    )


    samples=[]



    scale = (
        noise_level *
        np.max(
            np.abs(btc)
        )
    )


    for _ in range(
        n_bootstrap
    ):


        noise = np.random.normal(
            0,
            scale,
            size=btc.shape
        )


        btc_perturbed = (
            btc +
            noise
        )


        # concentration cannot be negative

        btc_perturbed = np.maximum(
            btc_perturbed,
            0
        )


        # normalize again

        integral = np.sum(
            btc_perturbed
        )


        if integral > 0:

            btc_perturbed /= integral



        H = recovery_function(
            btc_perturbed
        )


        samples.append(
            H
        )


    return np.asarray(
        samples
    )





def bootstrap_statistics(
        bootstrap_samples
):
    """
    Calculate mean and standard deviation
    from bootstrap ensemble.

    """

    samples=np.asarray(
        bootstrap_samples
    )


    mean=np.mean(
        samples,
        axis=0
    )


    std=np.std(
        samples,
        axis=0
    )


    return mean,std
