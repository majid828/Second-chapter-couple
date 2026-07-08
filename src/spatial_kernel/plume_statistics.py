"""
Spatial plume statistics.

Calculates:

mean position

variance

higher moments

"""

import numpy as np




def plume_centroid(
        x,
        G
):
    """
    Calculate plume centroid.

    mu1 = integral xG(x)dx

    """

    return np.trapz(
        x*G,
        x
    )





def plume_variance(
        x,
        G
):
    """
    Calculate spatial variance.

    sigma^2 =
    integral((x-mu)^2 G(x)dx)

    """

    mean=plume_centroid(
        x,
        G
    )


    variance=np.trapz(
        (
            x-mean
        )**2 *
        G,
        x
    )


    return variance





def plume_moments(
        x,
        G,
        order=4
):
    """
    Calculate central moments.

    """

    mean=plume_centroid(
        x,
        G
    )


    moments={}


    for n in range(
        1,
        order+1
    ):


        moments[n]=np.trapz(

            (
                x-mean
            )**n *
            G,

            x

        )


    return moments
