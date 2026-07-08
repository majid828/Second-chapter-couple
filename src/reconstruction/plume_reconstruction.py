"""
Forward plume reconstruction.

Implements:

C(x,t)=
M0 * integral_0^t
G(x,t-tau) H(tau)d tau


Inputs:

G(x,t)
H(t)

Output:

C(x,t)

"""

import numpy as np





def reconstruct_single_time(
        x,
        time_index,
        spatial_kernel,
        memory,
        dt,
        mass=1.0
):
    """
    Reconstruct concentration at one time.

    Parameters
    ----------
    x :
        spatial coordinate array


    time_index :
        current time index


    spatial_kernel :
        array:

        shape =
        (n_time,n_space)


    memory :
        H(t)


    dt :
        time step


    mass :
        injected mass


    Returns
    -------

    concentration profile

    """


    n_space=len(x)


    concentration=np.zeros(
        n_space
    )


    for j in range(
        time_index+1
    ):


        delay_index = (
            time_index-j
        )


        G_delay = (
            spatial_kernel[
                delay_index,
                :
            ]
        )


        concentration += (

            G_delay *

            memory[j] *

            dt

        )


    return mass*concentration





def reconstruct_plume(
        x,
        time,
        spatial_kernel,
        memory,
        mass=1.0
):
    """
    Reconstruct complete concentration field.


    Parameters
    ----------

    x :
        spatial grid


    time :
        time grid


    spatial_kernel :
        G(x,t)

        shape:

        (n_time,n_space)


    memory :
        H(t)


    mass :
        injected solute mass


    Returns
    -------

    C_pred:

        shape:

        (n_time,n_space)


    """


    x=np.asarray(x)

    time=np.asarray(time)

    spatial_kernel=np.asarray(
        spatial_kernel
    )

    memory=np.asarray(
        memory
    )



    n_time=len(time)

    n_space=len(x)



    if spatial_kernel.shape != (
        n_time,
        n_space
    ):

        raise ValueError(
            "Spatial kernel shape mismatch."
        )



    if len(memory)!=n_time:

        raise ValueError(
            "Memory length mismatch."
        )



    dt=time[1]-time[0]



    C_pred=np.zeros(
        (
            n_time,
            n_space
        )
    )


    for i in range(
        n_time
    ):


        C_pred[i,:]=reconstruct_single_time(

            x,

            i,

            spatial_kernel,

            memory,

            dt,

            mass

        )


    return C_pred
