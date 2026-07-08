"""
Tests for preprocessing module.
"""


import numpy as np
import pandas as pd
import tempfile
from pathlib import Path


from src.preprocessing.btc_processing import load_btc
from src.preprocessing.normalization import normalize_btc



def test_btc_loading():

    """
    Test BTC csv loading.
    """

    with tempfile.TemporaryDirectory() as tmp:

        file = Path(tmp) / "btc.csv"


        data = pd.DataFrame({

            "time":[0,1,2],

            "concentration":[0.0,1.0,0.5]

        })


        data.to_csv(
            file,
            index=False
        )


        time, concentration = load_btc(file)


        assert len(time)==3

        assert len(concentration)==3



def test_btc_normalization():

    """
    Test that normalized BTC integrates to one.
    """

    time=np.array(
        [0,1,2,3]
    )


    concentration=np.array(
        [0,1,2,1],
        dtype=float
    )


    normalized, area = normalize_btc(

        time,

        concentration

    )


    integral=np.trapz(

        normalized,

        time

    )


    assert np.isclose(

        integral,

        1.0,

        atol=1e-6

    )
