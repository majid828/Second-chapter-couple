"""
Field validation utilities.

Compares:

Observed tracer concentration

against

Reconstructed plume prediction.

"""

import numpy as np



def validate_field_prediction(
        observed,
        predicted
):
    """
    Calculate field prediction statistics.


    Returns:

    RMSE

    NRMSE

    MAE

    R2


    """

    observed=np.asarray(
        observed,
        dtype=float
    )

    predicted=np.asarray(
        predicted,
        dtype=float
    )



    if observed.shape != predicted.shape:

        raise ValueError(
            "Observed and predicted fields must have same shape."
        )



    error=observed-predicted



    rmse=np.sqrt(
        np.mean(
            error**2
        )
    )



    mae=np.mean(
        np.abs(error)
    )



    range_value=(

        np.max(observed)
        -
        np.min(observed)

    )


    if range_value>0:

        nrmse=rmse/range_value

    else:

        nrmse=np.nan



    ss_res=np.sum(
        error**2
    )


    ss_tot=np.sum(
        (
            observed-
            np.mean(observed)
        )**2
    )


    if ss_tot>0:

        r2=1-ss_res/ss_tot

    else:

        r2=np.nan



    return {

        "RMSE":rmse,

        "NRMSE":nrmse,

        "MAE":mae,

        "R2":r2

    }
