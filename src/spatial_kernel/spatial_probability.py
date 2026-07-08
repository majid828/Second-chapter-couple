"""
Spatial probability kernel reconstruction.

Implements:

G(x,t)=C(x,t)/integral(C(x,t)dx)

The concentration distribution is converted
into a probability density describing
solute location.

"""

import numpy as np




def spatial_probability_density(
        x,
        concentration
):
    """
    Convert concentration profile into
    spatial probability density.


    Parameters
    ----------
    x :
        spatial coordinate


    concentration :
        concentration at x


    Returns
    -------

    G :
        normalized spatial probability density


    """

    x=np.asarray(
        x,
        dtype=float
    )


    concentration=np.asarray(
        concentration,
        dtype=float
    )


    if len(x)!=len(concentration):

        raise ValueError(
            "x and concentration must have same length."
        )


    mass=np.trapz(
        concentration,
        x
    )


    if mass <= 0:

        raise ValueError(
            "Concentration integral must be positive."
        )


    G=(
        concentration /
        mass
    )


    return G





def spatial_mass(
        x,
        concentration
):
    """
    Calculate total spatial solute mass.

    """

    return np.trapz(
        concentration,
        x
    )
