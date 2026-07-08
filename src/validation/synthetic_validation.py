"""
Synthetic validation of memory recovery.

Purpose:

Generate known memory function:

H_true(t)

Generate synthetic BTC:

f = K H_true

Recover:

H_est

Evaluate recovery error.

"""

import numpy as np



def relative_l2_error(
        true,
        estimated
):
    """
    Relative L2 error:

    ||H-Hhat||2 / ||H||2

    """

    true=np.asarray(true)

    estimated=np.asarray(estimated)


    denominator=np.linalg.norm(
        true
    )


    if denominator==0:

        raise ValueError(
            "True vector cannot be zero."
        )


    return (

        np.linalg.norm(
            true-estimated
        )
        /
        denominator

    )





def validate_memory_recovery(
        true_memory,
        recovered_memory
):
    """
    Evaluate memory recovery.

    Returns:

    error

    correlation coefficient


    """

    true_memory=np.asarray(
        true_memory
    )

    recovered_memory=np.asarray(
        recovered_memory
    )


    error=relative_l2_error(

        true_memory,

        recovered_memory

    )


    correlation=np.corrcoef(

        true_memory,

        recovered_memory

    )[0,1]



    return {

        "relative_L2_error":
        error,


        "correlation":
        correlation

    }
