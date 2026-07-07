"""
Temporal memory function generator.

Implements:

1. Exponential memory

        H(t)=A exp(-beta t)


2. Power-law memory

        H(t)=A t^(-alpha)


3. Hybrid memory

        H(t)=A1 exp(-beta t)
             + A2 t^(-alpha)


The generated functions are normalized
over the finite observation window.

"""

import numpy as np



def _normalize_memory(time, memory):
    """
    Normalize memory over finite observation time.

    We intentionally avoid infinite-time
    normalization because fractional memory
    may not be integrable.

    """

    integral = np.trapz(
        memory,
        time
    )

    if integral <= 0:
        raise ValueError(
            "Memory integral must be positive."
        )

    return memory / integral




def exponential_memory(
        time,
        amplitude=1.0,
        beta=0.05
):
    """
    Generate exponential memory.

    H(t)=A exp(-beta t)

    Parameters
    ----------
    time :
        time array

    amplitude :
        scaling coefficient

    beta :
        decay rate


    Returns
    -------
    normalized memory

    """

    time = np.asarray(time)


    H = (
        amplitude *
        np.exp(
            -beta*time
        )
    )


    return _normalize_memory(
        time,
        H
    )





def power_law_memory(
        time,
        amplitude=1.0,
        alpha=0.5,
        epsilon=1e-6
):
    """
    Generate power-law memory.

    H(t)=A t^(-alpha)

    epsilon avoids singularity
    at t=0.

    """

    time = np.asarray(time)


    H = (
        amplitude *
        (time + epsilon)
        **(-alpha)
    )


    return _normalize_memory(
        time,
        H
    )





def hybrid_memory(
        time,
        A1=0.5,
        A2=0.5,
        beta=0.05,
        alpha=0.5
):
    """
    Generate hybrid memory.

    H(t)=A1 exp(-beta t)
        + A2 t^(-alpha)

    """

    time = np.asarray(time)


    exponential = (
        A1 *
        np.exp(
            -beta*time
        )
    )


    power = (
        A2 *
        (time+1e-6)
        **(-alpha)
    )


    H = exponential + power


    return _normalize_memory(
        time,
        H
    )
