""""
Script for computing the uncertainties by doing Bootstrapping.
"""
import numpy as np
import xarray as xr
import pandas as pd


def time_averaging_field(array, window=30, normalized=True):
    """Function averages a 3D field in a time window.
    Parameters
    ----------
    array: array
        3D array with dimensions (time, space, space). The time averaging
        happens in axis=0 of the array.
    window: int, optional
        The time window for the averaging. Default value is 30 (days).
    normalized: bool, optional
        Normalizes the average in space, axis=1&2. Default True.
    Returns
    -------
    averaged: array
        time averaged fields dimensions (time//window, space, space).
    time_array:
        1D array showing the window jumps. Its useless...
    """
    nt, nx, ny = array.shape

    new_t_dim = nt//window
    averaged = np.zeros((new_t_dim, nx, ny))
    time_array = np.arange(window, nt, window)

    for t in range(0, new_t_dim):
        index_slice = slice((t)*window, (t+1)*window)
        mean_aux = np.mean(array[index_slice, :, :], axis=0)

        if normalized:
            if mean_aux.sum() == 0:
                print(f'-- mean_aux.sum() = {mean_aux.sum()}')
                averaged[t] = np.zeros_like(mean_aux)
            else:
                averaged[t] = mean_aux/mean_aux.sum()

        else:
            averaged[t] = mean_aux

    # if normalized:
    #     print('--- Normalized?', averaged[-1].sum())

    return averaged, time_array


###############################################################################
# Setting the parameters
###############################################################################
series = 6  # the number of the simulation series
compute_mean = True  # True if you want to compute the average probability
average_window = 1234  # days (or stored time steps from parcels simulations)

# Bootstrap-parameters
sample_size = 100000
number_samples = 100  # at least 50 up to 100

print(f'Compute mean == {compute_mean}!')

domain_limits = [[-73, 25], [-80, 0]]
number_bins = (98, 80)  # defined with respect to domain_limits to be 1x1 deg

# generating the lon and lat ranges.
lon_range = np.linspace(domain_limits[0][0], domain_limits[0][1],
                        number_bins[0])
lat_range = np.linspace(domain_limits[1][0], domain_limits[1][1],
                        number_bins[1])

# Loading priors. Computed with release_points.py script.
# priors = pd.read_csv('../data/analysis/priors_river_inputs.csv', index_col=0)
priors = pd.read_csv('../priors_river_inputs.csv', index_col=0)
sources = list(priors.index)
number_sources = len(sources)

# Empty dictionaries to store computed probabilities.
likelihood = {}
posterior = {}
counts = {}
avg_label = ''  # label to be modified if average is True. Print in output_path

###############################################################################
# Building the histograms
###############################################################################
print('Building histograms')

time_dimensions = []

for loc in sources:
    print(f'- {loc}')
    # path_2_file = f"../data/simulations/sa-s{series:02d}" + \
    # f"/sa-s{series:02d}-{loc}.nc"
    path_2_file = "/data/oceanparcels/output_data/data_Claudio/" + \
        f"sa-s{series:02d}-{loc}.nc"
    particles = xr.load_dataset(path_2_file)

    trajectories = particles.dims['traj']
    time = particles.dims['obs']
    time_dimensions.append(time)  # to compute the minimum time between locs

    h = np.zeros((number_samples, time, *number_bins))
    # h_norm = np.zeros((time, *number_bins))
    for i_sample in range(number_samples):
        resampled_index = np.random.choice(trajectories, sample_size,
                                           replace=True)

        for t in range(time):
            lons = particles['lon'][resampled_index, t].values
            index = np.where(~np.isnan(lons))
            # lons = lons[index]
            lats = particles['lat'][resampled_index, t].values
            # lats = lats[index]

            if compute_mean:
                # if true, the histograms are not normalized.
                H, x_edges, y_edges = np.histogram2d(lons[index], lats[index],
                                                     bins=number_bins,
                                                     range=domain_limits)
                h[i_sample, t] = H

            else:
                # if false or else, the histograms are normalized, therefore
                # we get directly the likelihood.
                H_norm, x_edges, y_edges = np.histogram2d(lons[index],
                                                          lats[index],
                                                          bins=number_bins,
                                                          range=domain_limits,
                                                          density=True)
                h[i_sample, t] = H_norm

    counts[loc] = h
    #np.save('/scratch/cpierard/histograms_.npy', counts)
#    dump to npy file

# Some histograms have shorter time dimension. We select the shortest time
# of them all.
time = min(time_dimensions)

# Compute the total numer of particles per bin per time.
# total_counts = np.zeros((time, *number_bins))
# for loc in sources:
#     total_counts += counts[loc][:time]

###############################################################################
# To average or not to average, that's the question.
###############################################################################
if compute_mean:
    # we average the unnormalized histograms in a time window.
    print('Averaging histograms and computiong likelihood')

    for loc in sources:
        mean_samples = []

        for i_sample in range(number_samples):
            mean, time_range = time_averaging_field(counts[loc][i_sample],
                                                    window=average_window)
            mean_samples.append(mean)

        mean_samples = np.array(mean_samples)
        likelihood[loc] = mean_samples

    time = time//average_window
    avg_label = f'_aw{average_window}'  # average window nummer

###############################################################################
# Normalizing constant (sum of all hypothesis)
###############################################################################
print('Computing Normalizing constant')
normalizing_constant = np.zeros((number_samples, time, *number_bins))


for i_sample in range(number_samples):
    for t in range(time):
        # print('norm time', t)
        total = np.zeros((number_samples, number_sources, *number_bins))

        for i_sample in range(number_samples):
            for j, loc in enumerate(sources):
                total[i_sample, j] = likelihood[loc][i_sample, t]*priors['prior'][loc]

            normalizing_constant[i_sample, t] = np.sum(total[i_sample], axis=0)

###############################################################################
# Posterior probability
###############################################################################
print('Computing posterior probability')
likelihood_xr = {}  # formatting dictionary for xarray Dataset convertion
for k, loc in enumerate(sources):
    pst = np.zeros((number_samples, time, *number_bins))
    lklhd = np.zeros((number_samples, time, *number_bins))

    for i_sample in range(number_samples):
        for t in range(time):
            # Bayes theorem!
            pst[i_sample, t] = likelihood[loc][i_sample, t] * \
                priors['prior'][loc]/normalizing_constant[i_sample, t]
            lklhd[i_sample, t] = likelihood[loc][i_sample, t]
        # xarray Dataset formatting
        posterior[loc] = pst
        likelihood_xr[i_sample, loc] = lklhd

#####
# Standard deviation and mean
#####

mean = {}
standard_deviation = {}

for k, loc in enumerate(sources):
    mean[loc] = (["time", "x", "y"], np.mean(posterior[loc], axis=0))
    standard_deviation[loc] = (["time", "x", "y"],
                               np.std(posterior[loc], axis=0))

np.save(f'/scratch/cpierard/Means_{avg_label}.npy', standard_deviation)
np.save(f'/scratch/cpierard/Standard_deviation_{avg_label}.npy',
        standard_deviation)
###############################################################################
# Saving the likelihood & posteior as netCDFs
###############################################################################
coordinates = dict(time=time_range,
                   lon=(["x"], lon_range),
                   lat=(["y"], lat_range))

attributes = {'description': "Standard deviation Simulatons",
              'average_window': average_window}

# Posterior dataset
ds_post = xr.Dataset(data_vars=standard_deviation,
                     coords=coordinates,
                     attrs=attributes)

# output_path_post = f'../analysis/STD_{avg_label}.nc'
output_path_post = f'/scratch/cpierard/STD_{avg_label}_{series}_{number_samples}.nc'
# output_path_post = f'STD_{avg_label}.nc'

ds_post.to_netcdf(output_path_post)
ds_post.close()
