"""
Synthetic validation experiment.

Workflow:

1. Generate synthetic memory function H(t)
2. Generate normalized mobile kernel g(t)
3. Construct physical convolution matrix K
4. Generate BTC:

        f(t)=K H(t)

5. Add Gaussian measurement noise
6. Recover H(t) using constrained MAP inversion
7. Validate recovery
8. Save numerical results
9. Generate publication-quality figures


Run:

python experiments/synthetic/run_synthetic_experiment.py

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
    build_convolution_matrix,
    normalize_kernel
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


FIGURE_DIR = (
    RESULT_DIR /
    "figures"
)


TABLE_DIR = (
    RESULT_DIR /
    "tables"
)



for folder in [

    BTC_DIR,
    MEMORY_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    TABLE_DIR

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



    # ============================================
    # Time discretization
    # ============================================


    time = np.linspace(

        0,

        50,

        300

    )



    dt = np.mean(
        np.diff(time)
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


    # normalize true memory

    H_true = (

        H_true /

        np.trapezoid(
            H_true,
            time
        )

    )



    pd.DataFrame({

        "time":time,

        "true_memory":H_true

    }).to_csv(

        MEMORY_DIR /
        "true_memory.csv",

        index=False

    )



    # ============================================
    # Mobile transport kernel
    # ============================================


    g = mobile_kernel(

        time,

        m=2,

        b=0.05

    )


    # normalize mobile kernel

    g = normalize_kernel(

        time,

        g

    )



    # physical convolution matrix

    K = build_convolution_matrix(

        g,

        time

    )



    # ============================================
    # Generate BTC
    # ============================================


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



    # ============================================
    # Recover memory
    # ============================================


    recovered = recover_memory_kernel(

        time,

        btc_noisy,

        K,

        lam=1e-5,

        noise_variance=0.03**2

    )


    H_recovered = recovered[

        "memory"

    ]



    # normalize recovered memory

    H_recovered = np.maximum(

        H_recovered,

        0

    )


    H_recovered /= np.trapezoid(

        H_recovered,

        time

    )



    # ============================================
    # Validation
    # ============================================


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



    pd.DataFrame({

        "metric":
        list(metrics.keys()),

        "value":
        list(metrics.values())

    }).to_csv(

        TABLE_DIR /
        "recovery_metrics.csv",

        index=False

    )



    # ============================================
    # Save recovered kernel
    # ============================================


    pd.DataFrame({

        "time":time,

        "recovered_memory":
        H_recovered

    }).to_csv(

        RESULT_DIR /
        "recovered_memory.csv",

        index=False

    )



    # ============================================
    # Figure 1: BTC
    # ============================================


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(

        time,

        btc,

        label="Clean BTC"

    )


    plt.plot(

        time,

        btc_noisy,

        alpha=0.6,

        label="Noisy BTC"

    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Concentration"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /
        "synthetic_btc.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Figure 2: Memory recovery
    # ============================================


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(

        time,

        H_true,

        label="True H(t)"

    )


    plt.plot(

        time,

        H_recovered,

        "--",

        label="Recovered H(t)"

    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "Memory kernel"
    )


    plt.yscale(
        "log"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /
        "memory_recovery_log.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Figure 3: Mobile kernel
    # ============================================


    plt.figure(
        figsize=(8,5)
    )


    plt.plot(

        time,

        g

    )


    plt.xlabel(
        "Time"
    )


    plt.ylabel(
        "g(t)"
    )


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /
        "mobile_kernel.png",

        dpi=300

    )


    plt.close()



    # ============================================
    # Figure 4: Matrix
    # ============================================


    plt.figure(
        figsize=(6,5)
    )


    plt.imshow(

        K,

        aspect="auto"

    )


    plt.colorbar(
        label="K"
    )


    plt.xlabel(
        "Memory index"
    )


    plt.ylabel(
        "Time index"
    )


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR /
        "kernel_matrix.png",

        dpi=300

    )


    plt.close()



    print(
        "\nFigures saved:"
    )


    print(
        FIGURE_DIR
    )


    print(
        "\nSynthetic experiment completed."
    )



if __name__ == "__main__":

    main()
