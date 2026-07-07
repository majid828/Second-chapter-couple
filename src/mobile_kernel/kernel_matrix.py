"""
Construction of convolution matrix.

The inverse problem requires:

f = K H

where:

K = convolution matrix

H = memory function


This module converts the mobile kernel
into a lower triangular convolution matrix.

"""

import numpy as np




def build_convolution_matrix(
        mobile_kernel_values
):
    """
    Construct discrete convolution matrix.

    Parameters
    ----------

    mobile_kernel_values :
        normalized mobile kernel


    Returns
    -------

    K :
        convolution matrix


    """

    g = np.asarray(
        mobile_kernel_values,
        dtype=float
    )


    n = len(g)


    K = np.zeros(
        (n,n)
    )


    for i in range(n):

        for j in range(i+1):

            K[i,j] = g[i-j]



    return K





def check_convolution_matrix(
        K
):
    """
    Basic validation.

    A convolution matrix should be
    lower triangular.

    """

    upper = np.triu(
        K,
        k=1
    )


    return np.allclose(
        upper,
        0
    )
