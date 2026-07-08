"""
Error metrics for plume reconstruction.

Metrics:

RMSE

NRMSE

MAE

R2

"""

import numpy as np





def rmse(
        observed,
        predicted
):
    """
    Root mean square error.

    """

    observed=np.asarray(
        observed
    )

    predicted=np.asarray(
        predicted
    )


    return np.sqrt(
        np.mean(
            (
                observed-
                predicted
            )**2
        )
    )





def nrmse(
        observed,
        predicted
):
    """
    Normalized RMSE.

    NRMSE =
    RMSE/(max-min)


    """

    observed=np.asarray(
        observed
    )


    denominator=(

        np.max(observed)
        -
        np.min(observed)

    )


    if denominator==0:

        raise ValueError(
            "Observed range is zero."
        )


    return (
        rmse(
            observed,
            predicted
        )
        /
        denominator
    )





def mae(
        observed,
        predicted
):
    """
    Mean absolute error.

    """

    observed=np.asarray(
        observed
    )

    predicted=np.asarray(
        predicted
    )


    return np.mean(
        np.abs(
            observed-
            predicted
        )
    )





def r2_score(
        observed,
        predicted
):
    """
    Coefficient of determination.

    """

    observed=np.asarray(
        observed
    )

    predicted=np.asarray(
        predicted
    )


    ss_res=np.sum(
        (
            observed-
            predicted
        )**2
    )


    ss_tot=np.sum(
        (
            observed-
            np.mean(observed)
        )**2
    )


    if ss_tot==0:

        raise ValueError(
            "Observed data variance is zero."
        )


    return 1-(ss_res/ss_tot)





def reconstruction_metrics(
        observed,
        predicted
):
    """
    Calculate all reconstruction metrics.

    """

    return {

        "RMSE":
        rmse(
            observed,
            predicted
        ),


        "NRMSE":
        nrmse(
            observed,
            predicted
        ),


        "MAE":
        mae(
            observed,
            predicted
        ),


        "R2":
        r2_score(
            observed,
            predicted
        )

    }
