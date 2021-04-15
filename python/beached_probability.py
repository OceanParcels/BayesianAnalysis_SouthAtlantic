import numpy as np
import xarray as xr
import pandas as pd
import landmask as land
import modulo as mod

"""
Computes the probability field from a Ocean Parcels simulation.
Merges 2d likelihood_america of different OP output files.
1 output file per source.
"""


def average_field_coast(array, window=30):
    """It averages the counts_america and computes a probability map that adds
    up to 100%.
    """
    nt, ny = array.shape

    new_t_dim = nt//window
    averaged = np.zeros((new_t_dim, ny))
    time_array = np.array(range(1, new_t_dim))

    for t in range(0, new_t_dim):
        index_slice = slice((t)*window, (t+1)*window)
        mean_aux = np.mean(array[index_slice, :], axis=0)
        if mean_aux.sum() == 0:
            print(f'-- mean_aux.sum() = {mean_aux.sum()}')
            averaged[t] = np.zeros_like(mean_aux)
        else:
            averaged[t] = mean_aux/mean_aux.sum()

        print('-- Normalized?', averaged[t].sum())

    return averaged, time_array*window


# ###### Paramaters ########
# parameters for binning
series = 3

compute_mean = True

coarsen = True
factor = 10

average_window = 1500

landmask = np.load('../landmask.npy')
shore = land.get_shore_cells(landmask)
coast = land.get_coastal_cells(landmask)
lat_dim, lon_dim = landmask.shape

avg_label = ''
domain_limits = [[-73.0, 24.916666], [-79.916664, -5.0833335]]

lat_range = np.linspace(domain_limits[1][0], domain_limits[1][1],
                        lat_dim)

if coarsen:
    number_bins = (lat_dim + (factor - lat_dim % factor))//factor
else:
    number_bins = lat_dim

priors = pd.read_csv('../data/sources/river_inputs.csv', index_col=0)
# total_particles_beached = {}
counts_america = {}
counts_africa = {}
likelihood_america = {}
posterior_america = {}
likelihood_africa = {}
posterior_africa = {}
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


print('Building histograms')
for loc in sources:
    print(f'- {loc}')
    path_2_file = f"../data/simulations/sa-S03/sa-S03_{loc}.nc"
    particles = xr.load_dataset(path_2_file)
    n = particles.dims['traj']
    time = particles.dims['obs']

    h_ame = np.zeros((time, number_bins))
    h_afr = np.zeros((time, number_bins))
    # beached_loc = np.zeros(time)

    for t in range(time):
        lons = particles['lon'][:, t].values
        index = np.where(~np.isnan(lons))
        lons = lons[index]
        lats = particles['lat'][:, t].values
        index = np.where(~np.isnan(lats))
        lats = lats[index]

        H, x_edges, y_edges = np.histogram2d(lons, lats, bins=(1176, 899),
                                             range=domain_limits)

        H = np.nan_to_num(H)
        H = H*(coast.T + shore.T)
        # number_particles = H[0:500, :].sum()

        count_ame = np.sum(H[0:500, :], axis=0)
        count_afr = np.sum(H[900:1120, :], axis=0)

        if coarsen:
            h_ame[t], new_latitudes_ame = mod.coarsen_1D(count_ame, lat_range,
                                                         factor)
            h_afr[t], new_latitudes_afr = mod.coarsen_1D(count_afr, lat_range,
                                                         factor)
        else:
            h_ame[t] = count_ame
            h_afr[t] = count_afr

            new_latitudes = lat_range

        # beached_loc[t] = number_particles

    counts_america[loc] = h_ame
    counts_africa[loc] = h_afr
    # total_particles_beached[loc] = beached_loc

# creating parametes dictionary
parameter = {'domain_limits': domain_limits,
             'number_bins': number_bins,
             'lat_range_america': new_latitudes_ame,
             'lat_range_africa': new_latitudes_afr,
             'sources': sources}

################
if compute_mean:
    print('Averaging histograms and computiong likelihood_america.')
    avg_label = f'_average{average_window}'
    avg_likelihood_america = {}
    avg_likelihood_africa = {}
    for loc in sources:
        print(f'- {loc}')
        mean_ame, new_time_ame = average_field_coast(counts_america[loc],
                                                     window=average_window)
        mean_afr, new_time_afr = average_field_coast(counts_africa[loc],
                                                     window=average_window)

        avg_likelihood_america[loc] = mean_ame
        avg_likelihood_africa[loc] = mean_afr

    likelihood_america = avg_likelihood_america
    likelihood_africa = avg_likelihood_africa
    print('time same size', new_time_ame.shape == new_time_afr.shape)
    parameter['time_array'] = new_time_ame
    # parameter['time_array_africa'] = new_time_afr
    time = time//average_window

# Normalizing constant (sum of all hypothesis)
print('Computing Normailizing constant')
normalizing_constant = np.zeros((time, 2, number_bins))
# normalizing_constant_afr = np.zeros((time, 2, number_bins))

for t in range(time):
    total = np.zeros((number_sources, 2, number_bins))

    for j, loc in enumerate(sources):

        total[j, 0] = likelihood_america[loc][t]*priors['Mean'][loc]
        total[j, 1] = likelihood_africa[loc][t]*priors['Mean'][loc]

    normalizing_constant[t, 0] = np.sum(total[:, 0, :], axis=0)
    normalizing_constant[t, 1] = np.sum(total[:, 1, :], axis=0)

# posterior_america probability
print('Computing posterior probability')
for k, loc in enumerate(sources):
    aux_ame = np.zeros((time, number_bins))
    aux_afr = np.zeros((time, number_bins))

    for t in range(time):
        aux_ame[t] = likelihood_america[loc][t]*priors['Mean'][loc]/normalizing_constant[t, 0]
        aux_afr[t] = likelihood_africa[loc][t]*priors['Mean'][loc]/normalizing_constant[t, 1]
    posterior_america[loc] = aux_ame
    posterior_africa[loc] = aux_afr

# Saving the likelihood_america, posteior probabilityand parameters
# np.save(f'../data/analysis/sa-S{series:02d}/beach_posterior_america_sa-S{series:02d}{avg_label}.npy',
    # posterior_america, allow_pickle = True)

posterior = {'America': posterior_america,
             'Africa': posterior_africa}

np.save(f'../data/analysis/posterior.npy',
        posterior, allow_pickle=True)

np.save(f'../data/analysis/counts.npy',
        counts_africa, allow_pickle=True)


np.save(f'../data/analysis/params',
        parameter, allow_pickle=True)
