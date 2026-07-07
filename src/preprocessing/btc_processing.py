"""
Breakthrough curve processing utilities.

Input format:

CSV:

time, concentration


Output:

normalized BTC probability distribution.

"""

import pandas as pd
import numpy as np

from .normalization import normalize_btc



def load_btc(
        filepath
):
    """
    Load BTC csv file.

    Required columns:

    time
    concentration

    """

    data = pd.read_csv(filepath)


    required = [
        "time",
        "concentration"
    ]


    for col in required:
        if col not in data.columns:
            raise ValueError(
                f"Missing column: {col}"
            )


    return (
        data["time"].values,
        data["concentration"].values
    )




def process_btc(
        filepath
):
    """
    Complete BTC preprocessing.

    Returns:

    time
    raw concentration
    normalized BTC

    """

    time, concentration = load_btc(
        filepath
    )


    normalized, mass = normalize_btc(
        time,
        concentration
    )


    return {
        "time": time,
        "concentration": concentration,
        "normalized_btc": normalized,
        "mass": mass
    }
