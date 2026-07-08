"""
Noise sensitivity analysis for transport memory recovery.

This experiment evaluates robustness of the MAP inverse model
under different measurement noise levels.

Workflow:

1. Generate true memory kernel H(t)
2. Generate mobile transport kernel g(t)
3. Construct convolution matrix K
4. Generate clean BTC
5. Add different noise levels
6. Recover memory kernel
7. Calculate recovery metrics
8. Save tables and figures


Run:

python experiments/synthetic/run_noise_sensitivity.py

"""


import sys
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ------------------------------------------------
# Add project root
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT)
)



# ------------------------------------------------
# Import framework
# ------------------------------------------------


from src.synthetic.generate_memory import (
    hybrid_memory
)


from src.mobile_kernel import (
    mobile_kernel,
    normalize_kernel,
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
# Output folders
# ------------------------------------------------


RESULT_DIR = (
    ROOT /
    "results"
)


FIGURE_DIR = (
    RESULT_DIR /
    "figures"
)


TABLE_DIR = (
    RESULT_DIR /
    "tables"
)



for folder in [

    FIGURE_DIR,

    TABLE_DIR

]:

    folder.mkdir(

        parents=True,

        exist_ok=True

    )



# ------------------------------------------------
# Main experiment
# ------------------------------------------------


def main():


    print(
        "Starting noise sensitivity experiment..."
    )



    # ============================================
    # Time grid
    # ============================================


    time = np.linspace(

        0,

        50,

        300

    )



    # ============================================
    # True memory function
    # ============================================


    H_true = hybrid_memory(

        time,

        A1=0.4,

        A2=0.6,

        beta=0.08,

        alpha=0.8

    )



    H_true /= np.trapezoid(

        H_true,

        time

    )



    # ============================================
    # Mobile kernel
    # ============================================


    g = mobile_kernel(

        time,

        m=2,

        b=0.05

    )


    g = normalize_kernel(

        time,

        g

    )



    K = build_convolution_matrix(

        g,

        time

    )



    # ============================================
    # Generate clean BTC
    # ============================================


    btc_clean = K @ H_true



    # ============================================
    # Noise levels
    # ============================================


    noise_levels = [

        0.00,

        0.01,

        0.03,

        0.05,

        0.10,

        0.20

    ]



    results=[]



    # ============================================
    # Loop over noise
    # ============================================


    for noise in noise_levels:


        print(

            f"Running noise level {noise}"

        )



        btc_noisy = add_gaussian_noise(

            btc_clean,

            noise_level=noise,

            random_seed=42

        )



        recovered = recover_memory_kernel(

            time,

            btc_noisy,

            K,

            lam=1e-5,

            noise_variance=max(noise**2,1e-8)

        )


        H_rec = recovered[

            "memory"

        ]



        # physical correction

        H_rec = np.maximum(

            H_rec,

            0

        )


        H_rec /= np.trapezoid(

            H_rec,

            time

        )



        metrics = validate_memory_recovery(

            H_true,

            H_rec

        )



        results.append({

            "noise_level":

            noise,


            "relative_L2_error":

            metrics["relative_L2_error"],


            "correlation":

            metrics["correlation"]

        })



    # ============================================
    # Save table
    # ============================================


    df = pd.DataFrame(

        results

    )


    df.to_csv(

        TABLE_DIR /

        "noise_sensitivity_results.csv",

        index=False

    )



    print(df)



    # ============================================
    # Plot error
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


    plt.plot(

        df["noise_level"],

        df["relative_L2_error"],

        marker="o"

    )


    plt.xlabel(

        "Noise level"

    )


    plt.ylabel(

        "Relative L2 error"

    )


    plt.title(

        "Memory Recovery Error vs Noise"

    )


    plt.grid(True)


    plt.tight_layout()



    plt.savefig(

        FIGURE_DIR /

        "error_vs_noise.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Plot correlation
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


    plt.plot(

        df["noise_level"],

        df["correlation"],

        marker="o"

    )


    plt.xlabel(

        "Noise level"

    )


    plt.ylabel(

        "Correlation"

    )


    plt.ylim(

        0,

        1.05

    )


    plt.title(

        "Memory Recovery Correlation vs Noise"

    )


    plt.grid(True)


    plt.tight_layout()



    plt.savefig(

        FIGURE_DIR /

        "correlation_vs_noise.png",

        dpi=300

    )


    plt.close()



    print(

        "\nNoise sensitivity experiment completed."

    )


    print(

        "Results saved in:",

        TABLE_DIR

    )



if __name__ == "__main__":

    main()
