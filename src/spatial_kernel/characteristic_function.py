"""
Characteristic function analysis.

The characteristic function is:

phi(k)=integral exp(ikx)G(x)dx


It provides complete statistical
information about spatial transport.

"""

import numpy as np





def characteristic_function(
        x,
        G,
        k_values
):
    """
    Calculate spatial characteristic function.


    Parameters
    ----------

    x :
        spatial coordinate


    G :
        spatial probability density


    k_values :
        spatial frequencies


    Returns
    -------

    phi(k)


    """

    x=np.asarray(x)

    G=np.asarray(G)

    k_values=np.asarray(k_values)



    phi=np.zeros(
        len(k_values),
        dtype=complex
    )


    for i,k in enumerate(k_values):

        integrand = (
            np.exp(
                1j*k*x
            )
            *
            G
        )


        phi[i]=np.trapz(
            integrand,
            x
        )


    return phi





def inverse_characteristic_statistics(
        k_values,
        phi
):
    """
    Estimate variance from characteristic function.

    Near k=0:

    phi(k) ≈ 1 - sigma^2 k^2/2

    """

    k=np.asarray(k_values)

    phi=np.asarray(phi)


    real_phi=np.real(
        phi
    )


    # polynomial fit near zero

    coefficients=np.polyfit(
        k,
        real_phi,
        2
    )


    variance = (
        -2*
        coefficients[0]
    )


    return variance
