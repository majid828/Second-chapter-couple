"""
Synthetic validation experiment.

Complete workflow:

1. Generate synthetic memory function H(t)
2. Generate mobile kernel g(t)
3. Build convolution matrix K
4. Generate BTC:
        f = K H
5. Add measurement noise
6. Recover H using MAP inversion
7. Validate recovery
8. Save outputs


Run:

python experiments/synthetic/run_synthetic_experiment.py

"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------
# Add project root to Python path
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT)
)


# ------------------------------------------------
# Import framework modules
# ------------------------------------------------

from src.synthetic.generate_memory import (
    hybrid_memory
)

from src.mobile_kernel import (
    mobile_kernel,
    build_convolution_matrix
)

from src.synthetic.add_noise import (
    add_gaussian_noise
)

from src.inverse_model import (
    recover_memory_kernel
)

from src.validation import (
    validate_memory_recovery
)


# ------------------------------------------------
# Output directories
# ------------------------------------------------

SYNTHETIC_DIR = (

    ROOT /
    "data" /
    "synthetic"

)


BTC_DIR = (

    SYNTHETIC_DIR /
    "generated_btc"

)


MEMORY_DIR = (

    SYNTHETIC_DIR /
    "true_memory_functions"

)


RESULT_DIR = (

    ROOT /
    "results"

)



for folder in [

    BTC_DIR,

    MEMORY_DIR,

    RESULT_DIR

]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )



# ------------------------------------------------
# Main workflow
# ------------------------------------------------


def main():


    print(
        "Starting synthetic experiment..."
    )


    # ------------------------------
    # Time discretization
    # ------------------------------

    time = np.linspace(
        0,
        100,
        300
    )


    # ------------------------------
    # True memory function
    # ------------------------------

    H_true = hybrid_memory(

        time,

        A1=0.6,

        A2=0.4,

        beta=0.04,

        alpha=0.5

    )


    pd.DataFrame({

        "time":time,

        "true_memory":H_true

    }).to_csv(

        MEMORY_DIR /
        "true_memory.csv",

        index=False

    )



    # ------------------------------
    # Mobile transport kernel
    # ------------------------------

    g = mobile_kernel(

        time,

        m=2,

        b=0.05

    )


    K = build_convolution_matrix(
        g
    )



    # ------------------------------
    # Generate synthetic BTC
    # ------------------------------

    btc = K @ H_true



    btc_noisy = add_gaussian_noise(

        btc,

        noise_level=0.03,

        random_seed=42

    )



    pd.DataFrame({

        "time":time,

        "btc":btc,

        "btc_noisy":btc_noisy

    }).to_csv(

        BTC_DIR /
        "synthetic_btc.csv",

        index=False

    )



    # ------------------------------
    # Recover memory
    # ------------------------------

    recovered = recover_memory_kernel(

        time,

        btc_noisy,

        K,

        lam=1e-3,

        noise_variance=1e-4

    )


    H_recovered = recovered[
        "memory"
    ]



    # ------------------------------
    # Validation
    # ------------------------------

    metrics = validate_memory_recovery(

        H_true,

        H_recovered

    )


    print("\nRecovery Results")
    print("----------------")

    for key,value in metrics.items():

        print(
            f"{key}: {value:.6f}"
        )



    # ------------------------------
    # Save recovered kernel
    # ------------------------------

    pd.DataFrame({

        "time":time,

        "recovered_memory":
        H_recovered

    }).to_csv(

        RESULT_DIR /
        "recovered_memory.csv",

        index=False

    )


    print(
        "\nSynthetic experiment completed."
    )



if __name__ == "__main__":

    main()
