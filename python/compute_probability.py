import numpy as np
import xarray as xr
import pandas as pd

"""
Computes the probability field from a Ocean Parcels simulation.
Merges 2d likelihood of different OP output files.
1 output file per source.
"""
series = 3
compute_mean = True
average_window = 30


def average_field(array, window=30, normalized=True):
    nt, nx, ny = array.shape

    new_t_dim = nt//window
    averaged = np.zeros((new_t_dim, nx, ny))
    time_array = np.array(range(1, new_t_dim))

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

    return averaged, time_array*window


# ###### Paramaters ########
# parameters for binning
avg_label = ''
domain_limits = [[-73.0, 24.916666], [-79.916664, -5.0833335]]
number_bins = (120, 90)  # original from cmems is (1176, 899)
lon_range = np.linspace(domain_limits[0][0], domain_limits[0][1],
                        number_bins[0])
lat_range = np.linspace(domain_limits[1][0], domain_limits[1][1],
                        number_bins[1])


priors = pd.read_csv('../data/sources/river_inputs.csv', index_col=0)
likelihood = {}
posterior = {}
counts = {}
sources = ['Rio-de-Janeiro',
           'Rio-de-la-Plata',
           'Cape-Town',
           'Porto-Alegre',
           'Santos',
           'Cuvo',
           # 'Chiloango-Congo',
           'Luanda',
           'Itajai',
           'Paraiba']  # list(priors.keys())

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
    path_2_file = f"../data/simulations/sa-S03/sa-S03_{loc}.nc"
    particles = xr.load_dataset(path_2_file)
    n = particles.dims['traj']
    time = 1500  # particles.dims['obs']  # min_obs_lenght

    h = np.zeros((time, *number_bins))

    for t in range(time):
        lons = particles['lon'][:, t].values
        index = np.where(~np.isnan(lons))  # ugly statement
        lons = lons[index]
        lats = particles['lat'][:, t].values
        index = np.where(~np.isnan(lats))
        lats = lats[index]
        number_particles = len(lats)
        H, x_edges, y_edges = np.histogram2d(lons, lats, bins=number_bins,
                                             range=domain_limits)
        h[t] = H

    likelihood[loc] = h

################
if compute_mean:
    avg_label = f'_average{average_window}'
    avg_likelihood = {}
    for loc in sources:
        mean, new_time = average_field(likelihood[loc], window=average_window)
        mean_counts, trash = average_field(likelihood[loc],
                                           window=average_window,
                                           normalized=False)
        avg_likelihood[loc] = mean
        counts[loc] = mean_counts

    likelihood = avg_likelihood
    parameter['time_array'] = new_time
    time = time//average_window

# Normalizing constant (sum of all hypothesis)
normalizing_constant = np.zeros((time, *number_bins))

print('number sources', number_sources)
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
np.save(f'../data/analysis/sa-S{series:02d}/posterior_sa-S{series:02d}{avg_label}.npy',
        posterior, allow_pickle=True)
# np.save('../data/analysis/likelihood_sa-S03.npy',
#         likelihood, allow_pickle=True)
np.save(f'../data/analysis/sa-S{series:02d}/params_sa-S{series:02d}{avg_label}.npy',
        parameter, allow_pickle=True)

np.save(f'../data/analysis/sa-S{series:02d}/counts_sa-S{series:02d}{avg_label}.npy',
        counts, allow_pickle=True)
