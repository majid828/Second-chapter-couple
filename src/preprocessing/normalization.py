"""
Normalization functions for tracer transport data.

Implements:

BTC probability normalization:
    f(t)=C(t)/integral(C(t)dt)

Spatial probability normalization:
    G(x,t)=C(x,t)/integral(C(x,t)dx)

"""

import numpy as np


def normalize_btc(time, concentration):
    """
    Normalize breakthrough curve.

    Parameters
    ----------
    time : numpy.ndarray
        Observation time.

    concentration : numpy.ndarray
        BTC concentration values.

    Returns
    -------
    normalized_btc : numpy.ndarray
        Probability density representation.

    integral : float
        Original BTC mass integral.

    """

    time = np.asarray(time)
    concentration = np.asarray(concentration)


    integral = np.trapz(
        concentration,
        time
    )


    if integral <= 0:
        raise ValueError(
            "BTC integral must be positive."
        )


    normalized_btc = (
        concentration /
        integral
    )


    return normalized_btc, integral



def normalize_spatial_distribution(
        x,
        concentration
):
    """
    Convert spatial concentration field
    into probability density.

    Implements:

        G(x,t)=C(x,t)/integral(C(x,t)dx)


    Parameters
    ----------
    x : numpy.ndarray
        Spatial coordinate.

    concentration : numpy.ndarray
        Concentration distribution.


    Returns
    -------
    G : numpy.ndarray
        Spatial probability density.

    """

    x = np.asarray(x)
    concentration = np.asarray(concentration)


    integral = np.trapz(
        concentration,
        x
    )


    if integral <= 0:
        raise ValueError(
            "Spatial concentration integral must be positive."
        )


    G = concentration / integral


    return G



def finite_memory_normalization(
        time,
        memory
):
    """
    Normalize memory function over
    finite observation window.

    Important:
    We do NOT normalize over infinity because
    fractional memory functions may not be integrable.

    """

    time = np.asarray(time)
    memory = np.asarray(memory)


    integral = np.trapz(
        memory,
        time
    )


    if integral <= 0:
        raise ValueError(
            "Memory integral must be positive."
        )


    return memory / integral
