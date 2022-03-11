### Source Code for article "Attribution of Plastic Sources Using Bayesian Inference: Application to River-Sourced Floating Plastic in the South Atlantic Ocean"

This repository contains the code to reproduce the results shown in the article "Attribution of Plastic Sources Using Bayesian Inference: Application to River-Sourced Floating Plastic in the South Atlantic Ocean". In general, you can find the scripts for running the simulation and the scripts for analysing the output of the simulation.

This code uses [Ocean Parcels](https://oceanparcels.org) to perform the simulations. We recommend creating and environment to run the code as explained [here](https://oceanparcels.org/#installing). For the analysis, make sure to install the packages on the `requirements.txt`.

Here is a description of scripts according to it's function:

#### Analysis
To run the following scripts download data from the simulations from the supplementary material. For the scripts to work make sure to save the Data from the supplementary material within this repository such as `BayesianInference_SouthAtlantic/PierardBassottoMeirervanSebille_AttributionofPlastic`.

We recommend running the scripts in the following order:

1. `python/compute_probability.py`: computes the probabilities in the domain.
2. `python/beached_probability.py`: computes the probabilities
 for beached particles.
3. `python/article_plots.py`: plots the figures shown in the article.

#### Simulation
To run the simulations you need to download the SMOC velocity fields from April 2016 to august 2020.

1. Run before simulation:

    - `python/landmask.py`: creates the supporting fields that deal with the land-boundaries.
    - `python/release_positions.py`: creates the initial conditions for the particles.

2. Run Simulation:

    - `parcels_simulations.py`: script that runs the Lagrangian simulation.
    - `local_kernels.py`: contains the kernels used in the `parcels_simulations.py`. You don't need to run it.
