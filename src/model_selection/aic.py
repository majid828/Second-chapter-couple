"""
Akaike Information Criterion calculation.

Formula:

AIC =
N ln(RSS/N)+2k


Lower AIC indicates a better model
after penalizing complexity.

"""

import numpy as np





def calculate_aic(
        observed,
        predicted,
        number_of_parameters
):
    """
    Calculate Akaike Information Criterion.


    Parameters
    ----------

    observed :
        measured memory function


    predicted :
        model prediction


    number_of_parameters :
        number of fitted parameters


    Returns
    -------

    AIC value


    """

    observed=np.asarray(
        observed,
        dtype=float
    )

    predicted=np.asarray(
        predicted,
        dtype=float
    )


    if len(observed)!=len(predicted):

        raise ValueError(
            "Observed and predicted arrays must have same length."
        )


    N=len(observed)


    rss=np.sum(
        (
            observed-
            predicted
        )**2
    )


    if rss <=0:

        rss=1e-15



    aic=(

        N*
        np.log(
            rss/N
        )

        +

        2*
        number_of_parameters

    )


    return aic
