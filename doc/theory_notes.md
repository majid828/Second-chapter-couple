
# Transport Memory Kernel Framework: Theory Notes

## 1. Overview

The framework describes tracer transport using a memory-kernel formulation.
The observed breakthrough curve is represented as a convolution between a
mobile transport operator and a memory function.

## 2. Governing Concept

The transport response is written as:

C(t) = K H(t)

where:

- C(t) is the observed breakthrough response.
- K is the mobile transport kernel.
- H(t) is the temporal memory kernel.

## 3. Memory Models

Three physical memory models are considered:

### Exponential Memory

H(t)=A exp(-beta t)

Represents first-order exchange processes.

### Power-law Memory

H(t)=A(t+epsilon)^(-alpha)

Represents anomalous transport and long-time retention.

### Hybrid Memory

H(t)=A1 exp(-beta t)+A2(t+epsilon)^(-alpha)

Represents combined fast exchange and persistent memory.

## 4. Inverse Problem

The memory function is recovered using regularized Bayesian inversion.

The framework estimates:

- optimal memory kernel
- uncertainty bounds
- model confidence

## 5. Spatial Transport

Spatial concentration fields are converted into probability kernels:

G(x,t)=C(x,t)/Integral(C(x,t)dx)

Spatial statistics include:

- centroid
- variance
- skewness
- kurtosis

## 6. Multisite Classification

Recovered memory functions from multiple sites are compared using:

- Wasserstein distance
- Dynamic Time Warping
- Functional PCA
- clustering

This enables identification of transport memory classes.
