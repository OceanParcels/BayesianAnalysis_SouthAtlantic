"""
Computes the probability field of beached particles from Ocean Parcels
simulations. Computes the posterior probability in the latitude of the beached
particles.
"""
import numpy as np
import xarray as xr
import pandas as pd
import os


def time_averaging_coast(array, window=30):
    """It averages the counts_america and computes a probability map that adds
    up to 100%. It is built for the Beached particles 2D array.

    Parameters
    ----------
    array: array
        2D array with dimensions (time, space). The time averaging
        happens in axis=0 of the array.
    window: int, optional
        The time window for the averaging. Default value is 30 (days).
    normalized: bool, optional
        Normalizes the average in space, axis=1&2. Default True.

    Returns
    -------
    averaged: array
        time averaged fields dimensions (time//window, space).
    time_array:
        1D array showing the window jumps. Its useless...
    """
    nt, ny = array.shape

    new_t_dim = nt//window
    averaged = np.zeros((new_t_dim, ny))
    time_array = np.arange(window, nt, window)

    for t in range(0, new_t_dim):
        index_slice = slice((t)*window, (t+1)*window)
        mean_aux = np.mean(array[index_slice, :], axis=0)
        if mean_aux.sum() == 0:
            print(f'-- mean_aux.sum() = {mean_aux.sum()}')
            averaged[t] = np.zeros_like(mean_aux)
        else:
            averaged[t] = mean_aux/mean_aux.sum()

        print('-- Normalized?', averaged[t].sum())

    return averaged, time_array


# Creating the directory to store the analysis dataset
newpath = r'../analysis/'
if not os.path.exists(newpath):
    os.makedirs(newpath)

###############################################################################
# Setting the parameters
###############################################################################

compute_mean = True  # True if you want to compute the average probability
average_window = 1234  # window size for computing the probability

print(f'Compute mean == {compute_mean}!')

domain_limits = [[-73, 25], [-80, -5]]
number_bins = (98, 75)  # defined with respect to domain_limits to be 1x1 cell
half_point = number_bins[0]//2

lat_range = np.linspace(domain_limits[1][0], domain_limits[1][1],
                        number_bins[1])

# Loading priors. Computed with release_points.py script.
priors = pd.read_csv('../priors_river_inputs.csv',
                     index_col=0)
sources = list(priors.index)
number_sources = len(sources)

# Empty dictionaries to store computed probabilities.
counts_america = {}
counts_africa = {}
likelihood_america = {}
posterior_america = {}
likelihood_africa = {}
posterior_africa = {}
avg_label = ''

###############################################################################
# Building the histograms
###############################################################################
print('Building histograms')

time_dimensions = []
for loc in sources:
    print(f'- {loc}')
    path_2_file = f"../data/simulations/sa-s06/sa-s06-{loc}.nc"
    particles = xr.load_dataset(path_2_file)
    n = particles.dims['traj']
    time = particles.dims['obs']
    time_dimensions.append(time)

    # filter the particles that beached
    particles = particles.where((particles.beach == 1))

    h_ame = np.zeros((time, number_bins[1]))
    h_afr = np.zeros((time, number_bins[1]))
    # beached_loc = np.zeros(time)

    for t in range(time):
        lons = particles['lon'][:, t].values
        lats = particles['lat'][:, t].values
        index = np.where(~np.isnan(lons))
        lons = lons[index]
        lats = lats[index]

        # Compute the histogram
        H, x_edges, y_edges = np.histogram2d(lons, lats, bins=number_bins,
                                             range=domain_limits)

        H = np.nan_to_num(H)  # drop nans or covert them to zeros
        count_ame = np.sum(H[:55, :], axis=0)  # west meridional sum
        count_afr = np.sum(H[80:-5, :], axis=0)  # east meridional sum

        h_ame[t] = count_ame
        h_afr[t] = count_afr

    counts_america[loc] = h_ame
    counts_africa[loc] = h_afr

time = min(time_dimensions)
###############################################################################
# To average or not to average, that's the question.
###############################################################################
if compute_mean:
    print('Averaging histograms and computing likelihood')

    for loc in sources:
        print(f'- {loc}')
        mean_ame, time_range = time_averaging_coast(counts_america[loc],
                                                    window=average_window)
        mean_afr, _ = time_averaging_coast(counts_africa[loc],
                                           window=average_window)

        likelihood_america[loc] = mean_ame
        likelihood_africa[loc] = mean_afr

    time = time//average_window
    avg_label = f'average_{average_window}'

else:
    # convert counts to likelihood. The counts were normalized in line ~120.
    likelihood_america = counts_america
    likelihood_africa = counts_africa
    time_range = np.arange(0, time, 1)

###############################################################################
# Normalizing constant (sum of all hypothesis)
###############################################################################
print('Computing Normailizing constant')
normalizing_constant = np.zeros((time, 2, number_bins[1]))
# normalizing_constant_afr = np.zeros((time, 2, number_bins))

for t in range(time):
    total = np.zeros((number_sources, 2, number_bins[1]))

    for j, loc in enumerate(sources):

        total[j, 0] = likelihood_america[loc][t]*priors['prior'][loc]
        total[j, 1] = likelihood_africa[loc][t]*priors['prior'][loc]

    normalizing_constant[t, 0] = np.sum(total[:, 0, :], axis=0)
    normalizing_constant[t, 1] = np.sum(total[:, 1, :], axis=0)

###############################################################################
# Posterior probability
###############################################################################
print('Computing posterior probability')
for k, loc in enumerate(sources):
    aux_ame = np.zeros((time, number_bins[1]))
    aux_afr = np.zeros((time, number_bins[1]))

    for t in range(time):
        aux_ame[t] = likelihood_america[loc][t]*priors['prior'][loc] / \
            normalizing_constant[t, 0]
        aux_afr[t] = likelihood_africa[loc][t]*priors['prior'][loc] / \
            normalizing_constant[t, 1]
    posterior_america[loc] = (["time", "y"], aux_ame)
    posterior_africa[loc] = (["time",  "y"], aux_afr)

###############################################################################
# Saving the likelihood & posteior as netCDFs
###############################################################################
coordinates = dict(time=time_range,
                   lat=(["y"], lat_range))

attributes = {'description': "Beached posterior probability for America.",
              'average_window': average_window}
# Posterior dataset
post_ame = xr.Dataset(data_vars=posterior_america,
                      coords=coordinates,
                      attrs=attributes)

attributes = {'description': "Beached posterior probability for Africa.",
              'average_window': average_window}
# Posterior dataset
post_afr = xr.Dataset(data_vars=posterior_africa,
                      coords=coordinates,
                      attrs=attributes)

output_path_ame = newpath + f'beach_posterior_America_{avg_label}.nc'
output_path_afr = newpath + f'beach_posterior_Africa_{avg_label}.nc'

post_ame.to_netcdf(output_path_ame)
post_afr.to_netcdf(output_path_afr)
