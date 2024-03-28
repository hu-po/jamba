# Jamba

A minimal toy implementation of Mamba in JAX.

Jamba = JAX + Mamba

## Setup

This repo uses Docker containers for dependency management. To verify the setup, run the mnist example using `test.sh` script.

To run a training for the Jamba model use the `train.sh` script.

## Resources

Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf

Official GitHub implementation (Pytorch)
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

State Space Definitions
https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

Mamba Deep Dive
https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f

JAX Profiling:
https://jax.readthedocs.io/en/latest/profiling.html