"""
Optimization of mobile kernel parameters.

The objective is to estimate:

theta_g = (m,b)

such that:

f(t) ≈ g(t;theta_g)*H(t)

For this module we optimize the
mobile kernel shape using observed BTC
before memory inversion.

"""

import numpy as np

from scipy.optimize import minimize

from .parametric_kernel import (
    mobile_kernel
)





def objective_function(
        parameters,
        time,
        observed_btc
):
    """
    Objective function for mobile kernel.

    Parameters:

    parameters[0] = m

    parameters[1] = b


    """

    m, b = parameters



    # enforce physical constraints

    if m <= 0 or b <= 0:

        return 1e10



    g = mobile_kernel(
        time,
        m=m,
        b=b
    )


    error = np.sum(
        (
            observed_btc-g
        )**2
    )


    return error





def optimize_mobile_parameters(
        time,
        observed_btc,
        initial_guess=(2.0,0.05)
):
    """
    Estimate mobile kernel parameters.


    Parameters
    ----------

    time :
        time array


    observed_btc :
        normalized BTC


    initial_guess :
        starting values


    Returns
    -------

    dictionary containing:

    m

    b

    kernel

    optimization result


    """



    result = minimize(

        objective_function,

        x0=np.array(
            initial_guess,
            dtype=float
        ),

        args=(
            time,
            observed_btc
        ),

        method="Nelder-Mead"

    )


    m_opt, b_opt = result.x



    kernel = mobile_kernel(
        time,
        m=m_opt,
        b=b_opt
    )


    return {

        "m": m_opt,

        "b": b_opt,

        "kernel": kernel,

        "optimization_result": result

    }
