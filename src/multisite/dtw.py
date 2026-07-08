"""
Dynamic Time Warping similarity.

Used when two memory functions have
similar shapes but different time scales.

"""

import numpy as np





def dtw_distance(
        series1,
        series2
):
    """
    Calculate DTW distance.

    Parameters
    ----------

    series1 :

        memory function


    series2 :

        memory function



    Returns
    -------

    DTW distance


    """

    a=np.asarray(
        series1,
        dtype=float
    )


    b=np.asarray(
        series2,
        dtype=float
    )


    n=len(a)

    m=len(b)



    cost=np.zeros(
        (n+1,m+1)
    )


    cost[:,:]=np.inf


    cost[0,0]=0



    for i in range(1,n+1):

        for j in range(1,m+1):


            distance=abs(
                a[i-1]-b[j-1]
            )


            cost[i,j]=distance+min(

                cost[i-1,j],

                cost[i,j-1],

                cost[i-1,j-1]

            )


    return cost[n,m]





def dtw_distance_matrix(
        memory_functions
):
    """
    Pairwise DTW distance matrix.

    """

    H=np.asarray(
        memory_functions
    )


    n=len(H)


    matrix=np.zeros(
        (n,n)
    )


    for i in range(n):

        for j in range(i+1,n):


            d=dtw_distance(
                H[i],
                H[j]
            )


            matrix[i,j]=d

            matrix[j,i]=d


    return matrix
