"""
Fit physical memory models to recovered H(t).

Uses scipy nonlinear least squares.

Models:

- exponential
- power law
- hybrid


Returns fitted parameters
and predicted memory curve.

"""

import numpy as np

from scipy.optimize import curve_fit


from .exponential import (
    exponential_model
)

from .power_law import (
    power_law_model
)

from .hybrid_memory import (
    hybrid_model
)





def fit_memory_model(
        time,
        memory,
        model="hybrid"
):
    """
    Fit physical memory model.


    Parameters
    ----------

    time :
        time array


    memory :
        recovered H(t)


    model :
        exponential,
        power_law,
        hybrid



    Returns
    -------

    dictionary containing:

    parameters

    fitted curve

    covariance


    """

    time=np.asarray(
        time,
        dtype=float
    )


    memory=np.asarray(
        memory,
        dtype=float
    )



    if model=="exponential":


        function=exponential_model


        initial_guess=[

            1.0,

            0.05

        ]



    elif model=="power_law":


        function=power_law_model


        initial_guess=[

            1.0,

            0.5

        ]



    elif model=="hybrid":


        function=hybrid_model


        initial_guess=[

            0.5,

            0.05,

            0.5,

            0.5

        ]


    else:

        raise ValueError(
            "Unknown memory model."
        )



    bounds=(

        np.zeros(
            len(initial_guess)
        ),

        np.full(
            len(initial_guess),
            np.inf
        )

    )



    parameters,covariance = curve_fit(

        function,

        time,

        memory,

        p0=initial_guess,

        bounds=bounds,

        maxfev=50000

    )



    fitted=function(
        time,
        *parameters
    )



    return {

        "model":model,

        "parameters":parameters,

        "covariance":covariance,

        "fitted_memory":fitted

    }
