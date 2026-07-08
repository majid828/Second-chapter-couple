"""
Publication figure helpers.

Creates combined manuscript figures.

"""

import matplotlib.pyplot as plt





def save_publication_figure(
        filename,
        dpi=600
):
    """
    Save current matplotlib figure.

    Recommended:

    600 dpi for journals.

    """

    plt.savefig(

        filename,

        dpi=dpi,

        bbox_inches="tight"

    )
