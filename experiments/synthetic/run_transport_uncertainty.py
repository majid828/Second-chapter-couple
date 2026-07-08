"""
Transport kernel uncertainty analysis.

Purpose
-------
Evaluate robustness of memory kernel recovery when the
transport operator is uncertain.

Forward model:

    BTC = K_true H


Inverse model:

    BTC = K_est H


where:

    K_est != K_true


This represents uncertainty in:

- velocity
- dispersion
- transport parameters


Workflow
--------

1. Generate true memory function H(t)
2. Generate true mobile kernel g(t)
3. Construct K_true
4. Generate synthetic BTC
5. Perturb K_true to obtain K_est
6. Recover H using MAP inversion
7. Repeat Monte Carlo simulations
8. Compute statistics
9. Save tables and figures


Run
---

python experiments/synthetic/run_transport_uncertainty.py

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



for folder in [

    FIGURE_DIR,

    TABLE_DIR

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
        "Starting transport uncertainty experiment..."
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
    # True memory kernel
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
    # True transport kernel
    # ============================================


    g_true = mobile_kernel(

        time,

        m=2,

        b=0.05

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
    # Generate BTC
    # ============================================


    btc_clean = K_true @ H_true



    # ============================================
    # Transport uncertainty levels
    # ============================================

    uncertainty_levels = [

        0.00,

        0.01,

        0.02,

        0.05,

        0.10,

        0.20

    ]



    n_realizations = 20



    results = []



    # ============================================
    # Monte Carlo simulation
    # ============================================


    for uncertainty in uncertainty_levels:


        print(

            f"\nTransport uncertainty = {uncertainty}"

        )


        for realization in range(

            n_realizations

        ):



            np.random.seed(

                1000 +

                realization

            )



            # ------------------------------------
            # Perturb transport operator
            # ------------------------------------


            perturbation = (

                1 +

                uncertainty *

                np.random.randn(

                    *K_true.shape

                )

            )


            K_est = K_true * perturbation



            # physical constraint

            K_est = np.maximum(

                K_est,

                0

            )



            # ------------------------------------
            # Recover memory
            # ------------------------------------


            recovered = recover_memory_kernel(

                time,

                btc_clean,

                K_est,

                lam=1e-5,

                noise_variance=None

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
            # Validation
            # ------------------------------------


            metrics = validate_memory_recovery(

                H_true,

                H_rec

            )



            results.append({

                "transport_uncertainty":

                uncertainty,


                "realization":

                realization,


                "relative_L2_error":

                metrics["relative_L2_error"],


                "correlation":

                metrics["correlation"]

            })



    # ============================================
    # Save raw data
    # ============================================


    raw = pd.DataFrame(

        results

    )


    raw.to_csv(

        TABLE_DIR /

        "transport_uncertainty_raw.csv",

        index=False

    )



    # ============================================
    # Statistics
    # ============================================


    stats = (

        raw

        .groupby(

            "transport_uncertainty"

        )

        .agg(

            {

                "relative_L2_error":

                ["mean","std"],


                "correlation":

                ["mean","std"]

            }

        )

        .reset_index()

    )



    stats.columns = [

        "transport_uncertainty",

        "error_mean",

        "error_std",

        "correlation_mean",

        "correlation_std"

    ]



    stats.to_csv(

        TABLE_DIR /

        "transport_uncertainty_statistics.csv",

        index=False

    )



    print("\nResults")

    print(stats)



    # ============================================
    # Error figure
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


    plt.errorbar(

        stats["transport_uncertainty"],

        stats["error_mean"],

        yerr=stats["error_std"],

        marker="o"

    )


    plt.xlabel(

        "Transport uncertainty"

    )


    plt.ylabel(

        "Relative L2 error"

    )


    plt.title(

        "Effect of transport uncertainty on memory recovery"

    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "transport_uncertainty_error.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Correlation figure
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


    plt.errorbar(

        stats["transport_uncertainty"],

        stats["correlation_mean"],

        yerr=stats["correlation_std"],

        marker="o"

    )


    plt.xlabel(

        "Transport uncertainty"

    )


    plt.ylabel(

        "Correlation"

    )


    plt.ylim(

        0,

        1.05

    )


    plt.title(

        "Effect of transport uncertainty on recovery correlation"

    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "transport_uncertainty_correlation.png",

        dpi=300

    )


    plt.close()



    print(

        "\nTransport uncertainty experiment completed."

    )


    print(

        "Results saved in:",

        TABLE_DIR

    )



if __name__ == "__main__":

    main()
