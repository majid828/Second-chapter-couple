"""
Spatial concentration preprocessing.

Input CSV:

x,time,concentration


Creates:

G(x,t)

spatial probability distribution.

"""

import pandas as pd
import numpy as np

from .normalization import normalize_spatial_distribution




def load_spatial_data(
        filepath
):
    """
    Load spatial concentration data.

    Required columns:

    x
    time
    concentration

    """

    data = pd.read_csv(filepath)


    required = [
        "x",
        "time",
        "concentration"
    ]


    for col in required:
        if col not in data.columns:
            raise ValueError(
                f"Missing column {col}"
            )


    return data





def process_spatial_concentration(
        filepath,
        selected_time=None
):
    """
    Convert concentration snapshot
    into spatial probability kernel.


    Parameters
    ----------
    filepath :
        spatial csv file


    selected_time :
        optional observation time


    Returns
    -------

    x

    G(x,t)

    """

    data = load_spatial_data(
        filepath
    )


    if selected_time is not None:

        data = data[
            data["time"] ==
            selected_time
        ]


    x = data["x"].values

    concentration = (
        data["concentration"]
        .values
    )


    G = normalize_spatial_distribution(
        x,
        concentration
    )


    return {
        "x": x,
        "G": G,
        "time": selected_time
    }
