
# Joint Space-Time Memory Kernel Framework

This repository implements a synthetic benchmark for the proposed framework:

C(x,t) = G(x,t) * H(t)

where:
- G(x,t): spatial transport kernel
- H(t): temporal memory kernel

Workflow:
1. Generate synthetic BTC and plume snapshots.
2. Recover H(t) from BTC using regularized deconvolution.
3. Recover spatial plume distribution G(x,t) from snapshots using normalized concentration.
4. Reconstruct plume evolution.
5. Fit interpretable memory equations.

Run:
python run_pipeline.py

Note:
This is a research prototype. Kernel identifiability depends on assumptions and regularization.
