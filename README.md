### Source Code for article "Attribution of River-Sourced Floating Plastic in the South Atlantic Ocean Using Bayesian Inference"

This repository contains the scripts to reproduce the results shown in the article "Attribution of River-Sourced Floating Plastic in the South Atlantic Ocean Using Bayesian Inference". In general, you can find the scripts for running the simulation and the scripts for analysing the output of the simulation.

This code uses [Ocean Parcels](https://oceanparcels.org) to perform the simulations. We recommend creating and environment to run the code as explained [here](https://oceanparcels.org/#installing). for the analysis, make sure to install the packages on the `requirements.txt`.

Here is a description of scripts according to it's function.

#### Analysis
To run the following scripts download data from the simulations from the supplementary material.

- `python/compute_probability.py`: computes the probabilities in the domain.
- `python/beached_probability.py`: computes the probabilities
 for beached particles.
- `python/article_plots.py`: plots the figures shown in the article.

#### Simulation
1. Run before simulation:

    - `python/landmask.py`: creates the supporting fields that deal with the land-boundaries.

    - `python/release_positions.py`: creates the initial conditions for the particles.

2. Run Simulation:

    - `parcels_simulations.py`: script that runs the Lagrangian simulation.

    - `local_kernels.py`: contains the kernels used in the `parcels_simulations.py`. You don't need to run it.
