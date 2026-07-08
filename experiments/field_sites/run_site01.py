"""
Run complete analysis for Site 01.

Input:

data/raw/site_01/btc.csv


Output:

data/processed/
results/


Run:

python experiments/field_sites/run_site01.py

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------
# Project path
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT)
)


# ------------------------------------------------
# Imports
# ------------------------------------------------

from src.preprocessing.btc_processing import (
    load_btc
)

from src.preprocessing.normalization import (
    normalize_btc
)

from src.mobile_kernel import (
    mobile_kernel,
    build_convolution_matrix
)

from src.inverse_model import (
    recover_memory_kernel
)



SITE = "site_01"



def run_site(site_name):


    print(
        f"\nProcessing {site_name}"
    )


    raw_file = (

        ROOT /
        "data" /
        "raw" /
        site_name /
        "btc.csv"

    )


    if not raw_file.exists():

        raise FileNotFoundError(
            f"Missing file: {raw_file}"
        )



    # -----------------------------
    # Load BTC
    # -----------------------------

    time, concentration = load_btc(
        raw_file
    )



    # -----------------------------
    # Normalize BTC
    # -----------------------------

    btc_norm = normalize_btc(

        time,

        concentration

    )



    output_folder=(

        ROOT /
        "data" /
        "processed" /
        "normalized_btc"

    )


    output_folder.mkdir(

        parents=True,

        exist_ok=True

    )


    pd.DataFrame({

        "time":time,

        "normalized_concentration":
        btc_norm

    }).to_csv(

        output_folder /
        f"{site_name}_normalized_btc.csv",

        index=False

    )



    # -----------------------------
    # Mobile kernel
    # -----------------------------

    g = mobile_kernel(

        time,

        m=2,

        b=0.05

    )


    K = build_convolution_matrix(
        g
    )



    # -----------------------------
    # Bayesian inversion
    # -----------------------------

    result = recover_memory_kernel(

        time,

        btc_norm,

        K,

        lam=1e-3,

        noise_variance=1e-4

    )


    memory=result["memory"]



    kernel_folder=(

        ROOT /
        "data" /
        "processed" /
        "kernels"

    )


    kernel_folder.mkdir(

        parents=True,

        exist_ok=True

    )



    pd.DataFrame({

        "time":time,

        "memory_kernel":memory

    }).to_csv(

        kernel_folder /
        f"{site_name}_memory_kernel.csv",

        index=False

    )


    print(
        f"{site_name} completed successfully."
    )



if __name__=="__main__":

    run_site(
        SITE
    )
