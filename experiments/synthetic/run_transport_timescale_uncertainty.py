"""
Transport time-scale uncertainty analysis.

Purpose
-------
Evaluate robustness of memory kernel recovery when
the transport time scale is uncertain.

The mobile kernel is:

        g(t;m,b)

where:

        m : shape parameter
        b : transport decay/time-scale parameter


The forward model:

        BTC = K_true H


The inverse model:

        BTC = K_est H


where K_est is constructed using uncertain b.


Workflow
--------

1. Generate true memory function H(t)
2. Generate true mobile kernel
3. Construct K_true
4. Generate synthetic BTC
5. Perturb transport parameter b
6. Construct K_est
7. Recover memory kernel
8. Calculate recovery metrics
9. Perform Monte Carlo analysis
10. Save tables and figures


Run
---

python experiments/synthetic/run_transport_timescale_uncertainty.py

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
# Output directories
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
        "Starting transport time-scale uncertainty experiment..."
    )



    # ============================================
    # Time discretization
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
    # True transport parameters
    # ============================================


    m_true = 2.0

    b_true = 0.05



    # ============================================
    # True mobile kernel
    # ============================================


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
    # Generate BTC
    # ============================================


    btc_clean = K_true @ H_true



    # ============================================
    # Transport time-scale uncertainty
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



    results = []



    # ============================================
    # Monte Carlo loop
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
            # Perturb b parameter
            #
            # additive perturbation:
            #
            # b_est = b_true +/- uncertainty*b_true
            #
            # ------------------------------------


            b_est = b_true * (

                1 +

                uncertainty *

                np.random.randn()

            )



            # keep physical

            if b_est <= 0:

                b_est = b_true



            # ------------------------------------
            # Estimated kernel
            # ------------------------------------


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



            # physical correction

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



    # ============================================
    # Save raw results
    # ============================================


    raw = pd.DataFrame(

        results

    )


    raw.to_csv(

        TABLE_DIR /

        "transport_timescale_uncertainty_raw.csv",

        index=False

    )



    # ============================================
    # Statistics
    # ============================================


    stats = (

        raw

        .groupby(

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
    # Error plot
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


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


    plt.title(

        "Memory recovery sensitivity to transport time-scale uncertainty"

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
    # Correlation plot
    # ============================================


    plt.figure(

        figsize=(7,5)

    )


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


    plt.title(

        "Correlation degradation under transport uncertainty"

    )


    plt.grid(True)


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /

        "transport_timescale_uncertainty_correlation.png",

        dpi=300

    )


    plt.close()



    print(

        "\nTransport time-scale uncertainty experiment completed."

    )


    print(

        "Results saved in:",

        TABLE_DIR

    )



if __name__ == "__main__":

    main()
