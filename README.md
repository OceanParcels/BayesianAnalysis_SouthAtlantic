### Source Code for article "Attribution of River-Sourced Floating Plastic in the South Atlantic Ocean Using Bayesian Inference"

This repository contains the scripts to reproduce the results shown in the article "Attribution of River-Sourced Floating Plastic in the South Atlantic Ocean Using Bayesian Inference". In general, you can find the scripts for running the simulation and the scripts for analysing the output of the simulation.

This code uses [Ocean Parcels](https://oceanparcels.org) to perform the simulations. We recommend creating and environment to run the code as explained [here](https://oceanparcels.org/#installing). Also, you need to instal the `requirements.txt`

Here is a description of scripts according to it's function.

1. Run before simulation:

- `python/landmask.py`: creates the supporting fields that deal with the land-boundaries.

- `python/parcels_simulations.py`: creates the initial conditions for the particles.

2. Run Simulation:

- `parcels_simulations.py`: script that runs the Lagrangian simulation. But in order to run it, you need to generate the the support files by running - `python/landmask.py`.

- `local_kernels.py`: contains the kernels used in the `parcels_simulations.py`. You don't need to run it.

3. Analysis:
- `python/compute_probability.py`:
- `python/beached_probability.py`:
- `python/article_plots.py`:

All the scripts that deal with the probabilistic analysis are in the `python` directory.
