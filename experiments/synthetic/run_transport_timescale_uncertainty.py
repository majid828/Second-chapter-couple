"""
Transport time-scale uncertainty with measurement noise.

This experiment evaluates memory recovery robustness under
combined uncertainty:

1. Transport parameter uncertainty
2. Measurement noise

Forward model:

    BTC = K_true H + noise


Inverse model:

    BTC = K_est H


where:

    K_est is constructed using uncertain transport parameter b.


Run:

python experiments/synthetic/run_transport_timescale_uncertainty.py

"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# Project root
# ------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(ROOT)
)



# ------------------------------------------------
# Imports
# ------------------------------------------------


from src.synthetic.generate_memory import (
    hybrid_memory
)


from src.synthetic.add_noise import (
    add_gaussian_noise
)


from src.mobile_kernel import (
    mobile_kernel,
    normalize_kernel,
    build_convolution_matrix
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


RESULT_DIR = ROOT / "results"

FIGURE_DIR = RESULT_DIR / "figures"

TABLE_DIR = RESULT_DIR / "tables"

KERNEL_DIR = RESULT_DIR / "recovered_kernels"


for folder in [

    FIGURE_DIR,
    TABLE_DIR,
    KERNEL_DIR

]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )



# ------------------------------------------------
# Main
# ------------------------------------------------


def main():


    print(
        "Starting transport time-scale uncertainty experiment..."
    )



    # ============================================
    # Time
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



    pd.DataFrame({

        "time":time,

        "H_true":H_true

    }).to_csv(

        KERNEL_DIR /
        "H_true.csv",

        index=False

    )



    # ============================================
    # True transport parameters
    # ============================================

    m_true = 2.0

    b_true = 0.05



    g_true = mobile_kernel(

        time,

        m=m_true,

        b=b_true

    )


    g_true = normalize_kernel(

        time,

        g_true

    )



    K_true = build_convolution_matrix(

        g_true,

        time

    )



    # ============================================
    # Clean BTC
    # ============================================

    btc_clean = K_true @ H_true



    # ============================================
    # Uncertainty levels
    # ============================================

    uncertainty_levels = [

        0.00,

        0.05,

        0.10,

        0.20,

        0.30,

        0.40,

        0.50

    ]


    n_realizations = 20



    save_cases = {

        0.00,
        0.10,
        0.30,
        0.50

    }



    results = []

    saved = {}



    # ============================================
    # Monte Carlo
    # ============================================


    for uncertainty in uncertainty_levels:


        print(

            f"\nTime-scale uncertainty = {uncertainty}"

        )


        for realization in range(

            n_realizations

        ):



            np.random.seed(

                1000 +

                realization

            )



            # ------------------------------------
            # Transport uncertainty
            # ------------------------------------

            b_est = b_true * (

                1 +

                uncertainty *

                np.random.randn()

            )



            if b_est <= 0:

                b_est = b_true



            g_est = mobile_kernel(

                time,

                m=m_true,

                b=b_est

            )


            g_est = normalize_kernel(

                time,

                g_est

            )



            K_est = build_convolution_matrix(

                g_est,

                time

            )



            # ------------------------------------
            # Add measurement noise
            # ------------------------------------

            btc_observed = add_gaussian_noise(

                btc_clean,

                noise_level=0.05,

                random_seed=5000 + realization

            )



            # ------------------------------------
            # Recovery
            # ------------------------------------

            recovered = recover_memory_kernel(

                time,

                btc_observed,

                K_est,

                lam=1e-5,

                noise_variance=0.05**2

            )


            H_rec = recovered["memory"]



            H_rec = np.maximum(

                H_rec,

                0

            )


            integral = np.trapezoid(

                H_rec,

                time

            )


            if integral > 0:

                H_rec /= integral



            # ------------------------------------
            # Metrics
            # ------------------------------------

            metrics = validate_memory_recovery(

                H_true,

                H_rec

            )


            kernel_difference = (

                np.linalg.norm(

                    g_true-g_est

                )

                /

                np.linalg.norm(

                    g_true

                )

            )



            results.append({

                "timescale_uncertainty":

                uncertainty,

                "realization":

                realization,

                "estimated_b":

                b_est,

                "kernel_difference":

                kernel_difference,

                "relative_L2_error":

                metrics["relative_L2_error"],

                "correlation":

                metrics["correlation"]

            })



            # Save representative kernels

            if (

                uncertainty in save_cases

                and realization == 0

            ):


                saved[uncertainty] = H_rec.copy()



    # ============================================
    # Save recovered kernels
    # ============================================


    for level, kernel in saved.items():

        pd.DataFrame({

            "time":time,

            "memory":kernel

        }).to_csv(

            KERNEL_DIR /

            f"H_rec_{int(level*100)}percent.csv",

            index=False

        )



    # ============================================
    # Statistics
    # ============================================


    raw = pd.DataFrame(

        results

    )


    raw.to_csv(

        TABLE_DIR /

        "transport_timescale_uncertainty_raw.csv",

        index=False

    )



    stats = (

        raw.groupby(

            "timescale_uncertainty"

        )

        .agg(

            {

                "kernel_difference":

                ["mean","std"],


                "relative_L2_error":

                ["mean","std"],


                "correlation":

                ["mean","std"]

            }

        )

        .reset_index()

    )


    stats.columns = [

        "timescale_uncertainty",

        "kernel_difference_mean",

        "kernel_difference_std",

        "error_mean",

        "error_std",

        "correlation_mean",

        "correlation_std"

    ]


    stats.to_csv(

        TABLE_DIR /

        "transport_timescale_uncertainty_statistics.csv",

        index=False

    )



    print("\nResults")

    print(stats)



    # ============================================
    # Plot error
    # ============================================


    plt.figure(figsize=(7,5))


    plt.errorbar(

        stats["timescale_uncertainty"],

        stats["error_mean"],

        yerr=stats["error_std"],

        marker="o"

    )


    plt.xlabel(
        "Transport time-scale uncertainty"
    )


    plt.ylabel(
        "Relative L2 error"
    )


    plt.grid(True)

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "transport_timescale_uncertainty_error.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Plot correlation
    # ============================================


    plt.figure(figsize=(7,5))


    plt.errorbar(

        stats["timescale_uncertainty"],

        stats["correlation_mean"],

        yerr=stats["correlation_std"],

        marker="o"

    )


    plt.xlabel(
        "Transport time-scale uncertainty"
    )


    plt.ylabel(
        "Correlation"
    )


    plt.ylim(
        0,
        1.05
    )


    plt.grid(True)

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "transport_timescale_uncertainty_correlation.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Kernel comparison figure
    # ============================================


    plt.figure(figsize=(8,5))


    plt.plot(

        time,

        H_true,

        label="True"

    )


    for level, kernel in saved.items():

        plt.plot(

            time,

            kernel,

            label=f"{int(level*100)}%"

        )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Memory kernel H(t)"
    )


    plt.legend()


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "recovered_kernel_comparison.png",

        dpi=300

    )


    plt.close()



    print(

        "\nTransport time-scale uncertainty completed."

    )


    print(

        "Results saved in:",

        TABLE_DIR

    )



if __name__ == "__main__":

    main()
