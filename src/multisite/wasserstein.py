"""
Wasserstein distance calculation for memory functions.

Memory functions are treated as probability
distributions over time.

"""

import numpy as np

from scipy.stats import wasserstein_distance





def normalize_distribution(
        time,
        memory
):
    """
    Normalize memory function.

    """

    integral=np.trapz(
        memory,
        time
    )


    if integral<=0:

        raise ValueError(
            "Memory integral must be positive."
        )


    return memory/integral





def wasserstein_distance_matrix(
        time,
        memory_functions
):
    """
    Calculate pairwise Wasserstein distance.


    Parameters
    ----------

    time :
        common time grid


    memory_functions :

        array:

        (n_sites,n_time)



    Returns
    -------

    distance matrix


    """

    memory_functions=np.asarray(
        memory_functions,
        dtype=float
    )


    n_sites=memory_functions.shape[0]


    normalized=[]


    for H in memory_functions:

        normalized.append(
            normalize_distribution(
                time,
                H
            )
        )


    normalized=np.asarray(
        normalized
    )


    matrix=np.zeros(
        (n_sites,n_sites)
    )



    for i in range(n_sites):

        for j in range(i+1,n_sites):


            distance=wasserstein_distance(

                time,

                time,

                u_weights=normalized[i],

                v_weights=normalized[j]

            )


            matrix[i,j]=distance

            matrix[j,i]=distance



    return matrix
