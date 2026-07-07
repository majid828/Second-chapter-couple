"""
Parametric mobile transport kernel.

Implements:

g(t;m,b)
=
t^m exp(-b*t)
----------------
integral(t^m exp(-b*t)dt)


This represents the mobile travel-time
distribution used before memory convolution.

"""

import numpy as np



def normalize_kernel(
        time,
        kernel
):
    """
    Normalize kernel over finite observation window.

    Parameters
    ----------
    time :
        time array

    kernel :
        unnormalized kernel


    Returns
    -------
    normalized kernel

    """

    integral = np.trapz(
        kernel,
        time
    )


    if integral <= 0:

        raise ValueError(
            "Kernel integral must be positive."
        )


    return kernel / integral




def mobile_kernel(
        time,
        m=2.0,
        b=0.05,
        epsilon=1e-12
):
    """
    Generate parametric mobile transport kernel.


    Equation:

    g(t)=
    t^m exp(-bt)
    /
    integral(t^m exp(-bt)dt)


    Parameters
    ----------

    time :
        observation time array


    m :
        shape parameter


    b :
        decay parameter


    epsilon :
        avoids singularity at t=0


    Returns
    -------

    g :
        normalized mobile kernel


    """

    time = np.asarray(
        time,
        dtype=float
    )


    if np.any(time < 0):

        raise ValueError(
            "Time cannot contain negative values."
        )



    kernel = (

        (time + epsilon)**m *

        np.exp(
            -b*time
        )

    )


    return normalize_kernel(
        time,
        kernel
    )




def kernel_moments(
        time,
        kernel
):
    """
    Calculate first two moments
    of mobile kernel.

    Returns:

    mean travel time

    variance


    """

    mean = np.trapz(
        time*kernel,
        time
    )


    variance = np.trapz(
        (time-mean)**2 *
        kernel,
        time
    )


    return mean, variance
