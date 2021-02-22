import numpy as np
import xarray as xr
import pandas as pd

"""
Computes the probability field from a Ocean Parcels simulation.
Merges 2d likelihood of different OP output files.
1 output file per source.
"""
# ###### Paramaters ########

# parameters for binning
domain_limits = [[-73.0, 24.916666], [-79.916664, -5.0833335]]
number_bins = (120, 90)  # original from cmems is (1176, 899)
lon_range = np.linspace(domain_limits[0][0], domain_limits[0][1],
                        number_bins[0])
lat_range = np.linspace(domain_limits[1][0], domain_limits[1][1],
                        number_bins[1])




priors = pd.read_csv('../data/sources/river_inputs.csv', index_col=0)
likelihood = {}
posterior = {}
sources = ['Rio-de-Janeiro',
           'Rio-de-la-Plata',
           'Cape-Town',
           'Porto-Alegre',
           'Santos']  # list(priors.keys())

number_sources = len(sources)

# Storing the parameters of the simulations that are used for later
# processing of the data.
parameter = {'domain_limits': domain_limits,
             'number_bins': number_bins,
             'lon_range': lon_range,
             'lat_range': lat_range,
             'sources': sources}

for loc in sources:
    print(loc)
    path_2_file = f"../data/simulations/smoc/source_{loc}_K10_N100000.nc"
    particles = xr.load_dataset(path_2_file)
    n = particles.dims['traj']
    time = particles.dims['obs']
    h = np.zeros((time, *number_bins))

    for t in range(time):
        lons = particles['lon'][:, t].values
        index = np.where(np.isnan(lons) == False)  # ugly statement
        lons = lons[index]
        lats = particles['lat'][:, t].values
        index = np.where(np.isnan(lats) == False)
        lats = lats[index]
        number_particles = len(lats)
        H, x_edges, y_edges = np.histogram2d(lons, lats, bins=number_bins,
                                             range=domain_limits, density=True)
        h[t] = H

    likelihood[loc] = h

# Normalizing constant (sum of all hypothesis)
normalizing_constant = np.zeros((time, *number_bins))

for t in range(time):
    total = np.zeros((number_sources, *number_bins))

    for j, loc in enumerate(sources):
        total[j] = likelihood[loc][t]*priors['Mean'][loc]

    normalizing_constant[t] = np.sum(total, axis=0)

# Posterior probability
for k, loc in enumerate(sources):
    aux = np.zeros((time, *number_bins))

    for t in range(time):
        aux[t] = likelihood[loc][t]*priors['Mean'][loc]/normalizing_constant[t]

    posterior[loc] = aux

# Saving the likelihood, posteior probabilityand parameters
np.save('../data/analysis/posterior_smoc.npy', posterior, allow_pickle=True)
np.save('../data/analysis/likelihood_smoc.npy', likelihood, allow_pickle=True)
np.save('../data/analysis/params_smoc.npy', parameter, allow_pickle=True)
