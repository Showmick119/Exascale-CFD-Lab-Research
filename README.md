# Exascale Computational Fluid Dynamics Lab Research

### Goals:
- Low-latency **real-time inference** of flow fields in High Resolution CFD.
- **Accelerate** existing numerical solvers.

### Potential Approaches:
- Replacing the Runge-Kutta solver at some time-steps with a NN that predicts future flow states based on current state.
- Predict a few steps ahead and reduce total RK solves.
- Use NN to create predicted initial condition using natural language problem description.
- Train a model to learn fine-grid outputs from coarse-grid simulations.
- Upsample low-res CFD fields by training a Neural network on a ideal mapping similar to how NVIDIA DLSS works in taking a low-res render with motion vectors into a high-res image. 
- Using a coase computed grid, and provided physics parameters, be able to create fine-grid solution.
- Given past time steps, model can predict next mesh without solving PDEs.
- Train model to learn  how fields evolve over time, and given past ~10-20 frames, generate next frame.

### Architectures Explored:
- Physics Informed Neural Networks
- Operator Learning
- CNN-Based Super Resolution
- Fourier Neural Operator For Parametric Partial Differential Equations

### Tasks Completed So Far:
- Completed Standard C++ implementation of matrix operations and ported them to Kokkos.
- Read research papers and built intuition for how these CFD solvers work on deeper level, and how GPUs are leveraged and parallelized using tools like Kokkos.
