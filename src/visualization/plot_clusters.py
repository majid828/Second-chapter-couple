"""
Memory cluster visualization.

"""

import matplotlib.pyplot as plt





def plot_clusters(
        time,
        memory_functions,
        labels
):
    """
    Plot memory classes.


    Parameters

    time:
        common time grid


    memory_functions:

        (n_sites,n_time)


    labels:

        cluster IDs


    """

    plt.figure(
        figsize=(8,5)
    )



    for i,H in enumerate(memory_functions):


        plt.plot(

            time,

            H,

            label=f"Site {i+1} Class {labels[i]}"

        )



    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "H(t)"
    )


    plt.legend(
        fontsize=8
    )


    plt.tight_layout()


    plt.show()
