"""
Memory kernel visualization.

"""

import matplotlib.pyplot as plt





def plot_memory(
        time,
        memory,
        confidence=None
):
    """
    Plot recovered memory.


    confidence:

    tuple(lower,upper)

    """

    plt.figure(
        figsize=(7,4)
    )


    plt.plot(
        time,
        memory,
        label="Recovered H(t)"
    )



    if confidence is not None:


        lower,upper=confidence


        plt.fill_between(

            time,

            lower,

            upper,

            alpha=0.3,

            label="Uncertainty"

        )



    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Memory function"
    )


    plt.legend()

    plt.tight_layout()


    plt.show()
