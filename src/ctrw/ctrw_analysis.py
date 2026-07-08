"""
CTRW transport analysis.

This module connects recovered memory functions
with Continuous-Time Random Walk theory.

Main concepts:

Memory:

H(t)

Waiting-time distribution:

psi(t)


For anomalous transport:

psi(t) ~ t^(-1-alpha)


where:

alpha controls the long-time tail.

"""

import numpy as np

from scipy.stats import linregress





def estimate_power_law_exponent(
        time,
        memory,
        t_min=None
):
    """
    Estimate long-time power-law exponent.

    Assumes:

    H(t) ~ t^(-alpha)


    Uses logarithmic regression:

    log(H)= -alpha log(t)+C


    Parameters
    ----------

    time :
        time array


    memory :
        recovered memory function


    t_min :
        start point for tail fitting


    Returns
    -------

    alpha

    regression information


    """

    time=np.asarray(
        time,
        dtype=float
    )


    memory=np.asarray(
        memory,
        dtype=float
    )


    if len(time)!=len(memory):

        raise ValueError(
            "Time and memory lengths differ."
        )


    # Remove invalid values

    mask=(

        (time>0)

        &

        (memory>0)

    )


    time=time[mask]

    memory=memory[mask]



    # select long-time region

    if t_min is not None:

        mask=time>=t_min

        time=time[mask]

        memory=memory[mask]



    if len(time)<5:

        raise ValueError(
            "Not enough points for tail fitting."
        )


    log_t=np.log(
        time
    )


    log_H=np.log(
        memory
    )


    slope, intercept, r, p, stderr = linregress(

        log_t,

        log_H

    )


    alpha=-slope



    return {

        "alpha":alpha,

        "intercept":intercept,

        "r_squared":r**2,

        "stderr":stderr

    }





def generate_waiting_time_distribution(
        time,
        alpha,
        scale=1.0
):
    """
    Generate CTRW waiting-time distribution.


    Equation:

    psi(t)=A t^(-1-alpha)


    Parameters
    ----------

    time :
        time array


    alpha :
        CTRW exponent


    scale :
        amplitude



    Returns
    -------

    normalized psi(t)

    """

    time=np.asarray(
        time,
        dtype=float
    )


    if alpha<=0:

        raise ValueError(
            "Alpha must be positive."
        )


    psi=(

        scale *

        (time+1e-12)
        **(-1-alpha)

    )


    integral=np.trapz(
        psi,
        time
    )


    if integral<=0:

        raise ValueError(
            "Invalid waiting-time distribution."
        )


    psi/=integral


    return psi





def tail_statistics(
        time,
        memory,
        fraction=0.2
):
    """
    Calculate statistics of the long-time tail.

    The last fraction of observations
    is used.

    """

    time=np.asarray(
        time
    )

    memory=np.asarray(
        memory
    )


    n=len(time)


    start=int(
        (1-fraction)*n
    )


    tail_memory=memory[start:]

    tail_time=time[start:]


    slope_data=estimate_power_law_exponent(

        tail_time,

        tail_memory

    )


    return slope_data





def ctrw_memory_comparison(
        time,
        memory
):
    """
    Compare recovered memory with CTRW theory.


    Returns:

    - estimated alpha
    - CTRW waiting distribution
    - goodness of power-law fit


    """

    result=estimate_power_law_exponent(

        time,

        memory

    )


    alpha=result["alpha"]


    psi=generate_waiting_time_distribution(

        time,

        alpha

    )


    return {

        "alpha":alpha,

        "power_law_R2":
        result["r_squared"],

        "waiting_time_distribution":
        psi

    }
