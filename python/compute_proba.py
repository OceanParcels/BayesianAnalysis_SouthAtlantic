import my_module as utils
import numpy as np
import pickle
import xarray as xr
import itertools

"""
Computes the probability field from a Ocean Parcels simulation.

Merges 2d histograms of different OP output files.
1 output file per source.
"""
# ###### Paramaters #############
# Loading data
# Plastic sources
infile = open('../river_sources.pkl', 'rb')
river_sources = pickle.load(infile)
# Labels of the sources.
labels = ['Rio-de-Janeiro', 'Rio-de-la-Plata']  # list(river_sources.keys())
domain_coords = utils.domain_coords

for loc in labels:
    path_2_file = f"../data/source_{loc}_release.nc"
    particles = xr.load_dataset(path_2_file)

    n = particles.dims['traj']
    time = particles.dims['obs']

    n = particles.dims['traj']
    time = particles.dims['obs']
    coarse_hist = {}
    dimensions = {}
    bin_factor = 15
    h = []

    for t in range(time):
        lons = particles['lon'][:, t].values
        index = np.where(np.isnan(lons) is False)
        lons = lons[index]
        lats = particles['lat'][:, t].values
        index = np.where(np.isnan(lats) is False)
        lats = lats[index]
        number_particles = sum(np.isnan(lats) == False)

        # appending corner points of the domain to keep the 2d histograms
        # the same dimension
        print(domain_coords['lat_lims'], domain_coords['lon_lims'])

        for coord_tuple in itertools.product(domain_coords['lat_lims'],
                                             domain_coords['lon_lims']):

            lats = np.hstack((lats, coord_tuple[0]))
            lons = np.hstack((lons, coord_tuple[1]))

        H, Lo, La = np.histogram2d(lons, lats, bins=(1176, 899))
        # h_mask = np.ma.masked_array(H, mask=mask.transpose())

        # np.vstack((h,H))
        # computing probability
        h_coarse = utils.coarsen(H, domain_coords['lon'], domain_coords['lat'],
                                 bin_factor)
        h.append(h_coarse[0])  # just storing the values

    coarse_hist[loc] = np.array(h)
    # dimensions['mask'] = h_coarse[0].mask
# dimensions['lat'] = h_coarse[2]
# dimensions['lon'] = h_coarse[1]
# coarse_hist['dimensions'] = dimensions


np.save('temp.npy', coarse_hist)

# with open('../data/probabilities_rmv_grounded.pkl', 'wb') as f:
#     pickle.dump(coarse_hist, f)
