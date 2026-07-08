"""
BTC visualization.

"""

import matplotlib.pyplot as plt





def plot_btc(
        time,
        concentration,
        normalized=None
):
    """
    Plot breakthrough curve.

    """

    plt.figure(
        figsize=(7,4)
    )


    plt.plot(
        time,
        concentration,
        label="BTC"
    )


    if normalized is not None:

        plt.plot(
            time,
            normalized,
            label="Normalized BTC"
        )


    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Concentration"
    )


    plt.legend()

    plt.tight_layout()


    plt.show()
