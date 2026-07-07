# Physics-Constrained Transport Memory Kernel Framework

## Overview

This repository implements a physics-constrained inverse framework for
recovering transport memory functions from tracer breakthrough curves
and spatial plume observations.

The framework separates:

1. Spatial transport behavior:

\[
G(x,t)
\]

2. Temporal memory behavior:

\[
H(t)
\]


The complete workflow is:


## Scientific Framework

The concentration field is represented as:

\[
C(x,t)=M_0\int_0^t G(x,t-\tau)H(\tau)d\tau
\]


where:

- G(x,t): spatial transport probability density
- H(t): temporal transport memory function


## Repository Structure
