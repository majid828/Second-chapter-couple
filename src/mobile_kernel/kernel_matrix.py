"""
Construction of physically consistent convolution matrix.

Forward model:

f(t)=∫ g(t-τ)H(τ)dτ


Discrete form:

f_i = Σ_j K_ij H_j Δt


where:

K_ij = g(t_i-t_j)

"""


import numpy as np



def build_convolution_matrix(
        mobile_kernel_values,
        time=None
):
    """
    Construct causal convolution matrix.

    Parameters
    ----------
    mobile_kernel_values :
        mobile transport kernel g(t)


    time :
        time vector

    Returns
    -------

    K :
        convolution matrix including dt scaling

    """



    g=np.asarray(
        mobile_kernel_values,
        dtype=float
    )


    n=len(g)



    # -----------------------------
    # Time step
    # -----------------------------

    if time is not None:

        time=np.asarray(
            time,
            dtype=float
        )

        dt=np.mean(
            np.diff(time)
        )


    else:

        dt=1.0



    # -----------------------------
    # Convolution matrix
    # -----------------------------


    K=np.zeros(
        (n,n)
    )



    for i in range(n):

        for j in range(i+1):

            K[i,j]=g[i-j]*dt



    return K




def normalize_kernel(
        time,
        kernel
):
    """
    Normalize transport kernel:

    ∫g(t)dt = 1

    """

    time=np.asarray(time)

    kernel=np.asarray(kernel)


    area=np.trapezoid(
        kernel,
        time
    )


    if area<=0:

        raise ValueError(
            "Kernel integral must be positive."
        )


    return kernel/area




def check_convolution_matrix(
        K
):
    """
    Check causal lower triangular structure.
    """

    upper=np.triu(
        K,
        k=1
    )


    return np.allclose(
        upper,
        0
    )
