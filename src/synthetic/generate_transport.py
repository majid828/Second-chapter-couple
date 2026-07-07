"""
Synthetic spatial transport generator.

Implements Gaussian advection-dispersion
transport kernel:

G(x,t)=
1/sqrt(4*pi*D*t)
*
exp(-(x-vt)^2/(4Dt))


and generates:

C(x,t)=
integral G(x,t-tau)H(tau)d tau

"""

import numpy as np





def gaussian_transport_kernel(
        x,
        time,
        velocity=1.0,
        dispersion=0.1
):
    """
    Generate spatial transport kernel.

    Parameters
    ----------

    x :
        spatial coordinates

    time :
        time value

    velocity :
        pore velocity

    dispersion :
        dispersion coefficient


    Returns
    -------

    G(x,t)

    """

    x = np.asarray(x)


    if time <= 0:

        return np.zeros_like(x)



    coefficient = (
        1.0 /
        np.sqrt(
            4*np.pi*
            dispersion*
            time
        )
    )


    exponent = (
        -(x-velocity*time)**2
        /
        (
            4*
            dispersion*
            time
        )
    )


    G = (
        coefficient *
        np.exp(exponent)
    )


    # probability normalization

    integral = np.trapz(
        G,
        x
    )


    if integral > 0:

        G = G/integral


    return G






def generate_concentration_field(
        x,
        time,
        memory_function,
        velocity=1.0,
        dispersion=0.1
):
    """
    Generate synthetic concentration field.

    Implements:

    C(x,t)=
    integral_0^t
    G(x,t-tau)H(tau)d tau


    Parameters
    ----------

    x :
        spatial grid

    time :
        temporal grid

    memory_function :
        array H(t)


    Returns
    -------

    concentration matrix

    """

    x = np.asarray(x)
    time = np.asarray(time)


    n_time = len(time)
    n_space = len(x)


    C = np.zeros(
        (n_time,n_space)
    )


    dt = time[1]-time[0]


    for i,t in enumerate(time):


        concentration = np.zeros(
            n_space
        )


        for j,tau in enumerate(time[:i+1]):


            delay = t-tau


            G = gaussian_transport_kernel(
                x,
                delay,
                velocity,
                dispersion
            )


            concentration += (
                G *
                memory_function[j]
                *
                dt
            )


        C[i,:] = concentration



    return C
