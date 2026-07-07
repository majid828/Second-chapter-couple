"""
Physical constraints for memory recovery.

The recovered memory function must satisfy:

1. Positivity:

H(t)>=0


2. Finite observation normalization:

integral H(t)dt = 1

"""

import numpy as np




def positivity_constraint(
        H
):
    """
    Check positivity constraint.

    Parameters
    ----------
    H :
        memory vector


    Returns
    -------
    bool

    """

    H=np.asarray(H)

    return np.all(
        H >= 0
    )





def normalization_constraint(
        time,
        H,
        tolerance=1e-6
):
    """
    Check finite-window normalization.

    integral_0^T H(t)dt = 1


    """

    integral=np.trapz(
        H,
        time
    )


    return abs(
        integral-1.0
    ) < tolerance




def normalize_memory(
        time,
        H
):
    """
    Normalize recovered memory.

    Used after unconstrained operations.

    """

    integral=np.trapz(
        H,
        time
    )


    if integral <=0:

        raise ValueError(
            "Memory integral must be positive."
        )


    return H/integral
