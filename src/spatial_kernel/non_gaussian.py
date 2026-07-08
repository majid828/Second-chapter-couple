"""
Non-Gaussian plume characterization.

Calculates:

Skewness:

S = mu3/sigma^3


Kurtosis:

K = mu4/sigma^4


Gaussian distribution:

S=0

K=3

"""

import numpy as np

from .plume_statistics import (
    plume_moments,
    plume_variance
)





def skewness(
        x,
        G
):
    """
    Calculate plume skewness.

    """

    moments=plume_moments(
        x,
        G,
        order=3
    )


    variance=plume_variance(
        x,
        G
    )


    if variance<=0:

        raise ValueError(
            "Variance must be positive."
        )


    sigma=np.sqrt(
        variance
    )


    return (
        moments[3]
        /
        sigma**3
    )





def kurtosis(
        x,
        G
):
    """
    Calculate kurtosis.

    Gaussian value = 3

    """

    moments=plume_moments(
        x,
        G,
        order=4
    )


    variance=plume_variance(
        x,
        G
    )


    if variance<=0:

        raise ValueError(
            "Variance must be positive."
        )


    return (
        moments[4]
        /
        variance**2
    )





def non_gaussian_metrics(
        x,
        G
):
    """
    Return complete non-Gaussian metrics.

    """

    return {

        "centroid":
        np.trapz(
            x*G,
            x
        ),

        "variance":
        plume_variance(
            x,
            G
        ),

        "skewness":
        skewness(
            x,
            G
        ),

        "kurtosis":
        kurtosis(
            x,
            G
        )

    }
