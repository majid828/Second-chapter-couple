"""
Bayesian Information Criterion.

Formula:

BIC =
N ln(RSS/N)+k ln(N)


BIC penalizes model complexity
more strongly than AIC.

"""

import numpy as np





def calculate_bic(
        observed,
        predicted,
        number_of_parameters
):
    """
    Calculate Bayesian Information Criterion.


    Parameters
    ----------

    observed :
        observed memory


    predicted :
        fitted memory


    number_of_parameters :
        number of fitted parameters


    Returns
    -------

    BIC value


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
            "Array length mismatch."
        )


    N=len(observed)


    rss=np.sum(
        (
            observed-
            predicted
        )**2
    )


    if rss<=0:

        rss=1e-15



    bic=(

        N*
        np.log(
            rss/N
        )

        +

        number_of_parameters*
        np.log(N)

    )


    return bic
