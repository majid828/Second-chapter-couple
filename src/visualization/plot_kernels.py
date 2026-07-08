"""
Kernel visualization.

"""

import matplotlib.pyplot as plt





def plot_kernel(
        x,
        kernel,
        label="Kernel"
):
    """
    Plot transport kernel.

    """

    plt.figure(
        figsize=(7,4)
    )


    plt.plot(
        x,
        kernel,
        label=label
    )


    plt.xlabel(
        "Position"
    )


    plt.ylabel(
        "Probability density"
    )


    plt.legend()


    plt.tight_layout()


    plt.show()
