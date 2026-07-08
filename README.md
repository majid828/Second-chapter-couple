# Physics-Constrained Transport Memory Kernel Framework

A reproducible computational framework for identifying, recovering, and classifying transport memory kernels from tracer breakthrough curves.

The framework combines:

- physics-based transport kernels
- Bayesian inverse modeling
- uncertainty quantification
- spatial plume analysis
- memory mechanism identification
- multisite transport classification


## Overview

Transport in heterogeneous porous media often exhibits non-Fickian behavior due to:

- matrix diffusion
- stagnant zones
- preferential pathways
- multiscale velocity distributions

This framework introduces a memory-kernel approach where the tracer response is represented as:

\[
C(t)=K H(t)
\]

where:

- \(C(t)\) is the observed breakthrough curve
- \(K\) is the mobile transport operator
- \(H(t)\) is the transport memory kernel


## Main Features

### 1. BTC Processing

- tracer breakthrough curve loading
- normalization
- spatial concentration processing


### 2. Transport Kernel Modeling

Includes:

- exponential kernels
- power-law kernels
- hybrid memory kernels


### 3. Inverse Memory Recovery

The framework estimates:

- temporal memory kernels
- transport parameters
- uncertainty bounds


### 4. Uncertainty Quantification

Methods:

- Bayesian posterior covariance
- bootstrap uncertainty analysis


### 5. Spatial Transport Analysis

Includes:

- spatial probability kernels
- plume statistics
- characteristic functions
- non-Gaussian transport indicators


### 6. Multisite Classification

Transport memory behavior is compared using:

- Wasserstein distance
- Dynamic Time Warping
- Functional PCA
- clustering


# Repository Structure
