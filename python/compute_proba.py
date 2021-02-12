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
# Loading data
# Plastic sources
infile = open('../river_sources.pkl', 'rb')
river_sources = pickle.load(infile)
# Labels of the sources.
labels = list(river_sources.keys())

# parcels trajectories (big file 30 GB)
particles = xr.load_dataset('../data/forward_2years.nc')

domain_coords = utils.domain_coords
print(domain_coords)

n = 100000
time = particles['lon'].shape[1]

coarse_hist = {}
dimensions = {}

for j, n in enumerate(range(0, 1000000, 100000)):
    m = n+100000
    h = []
    for t in range(120, 1321, 200):
        print(labels[j], t)
        lons = particles['lon'][n:m, t].values
        index = np.where(np.isnan(lons) is False)
        lons = lons[index]
        lats = particles['lat'][n:m, t].values
        index = np.where(np.isnan(lats) is False)
        lats = lats[index]
        number_particles = sum(np.isnan(lats) is False)
        # print(number_particles)

        # apppending corner points of the domain to keep the 2d histograms the same dimension
        for coord_tuple in itertools.product(domain_coords['lat_lims'], domain_coords['lon_lims']):
            lats = np.hstack((lats, coord_tuple[0]))
            lons = np.hstack((lons, coord_tuple[1]))

        H, Lo, La = np.histogram2d(lons, lats, bins=(1176, 899))
        h_mask = np.ma.masked_array(H, mask=domain_coords['land_mask'].transpose())
        #histograms[labels[j]] = h_mask

        # computing probability
        h_coarse = utils.coarsen(h_mask, domain_coords['lon'], domain_coords[' lat'], 15)
        h.append(h_coarse[0].data)  # just storing the values

    coarse_hist[labels[j]] = np.array(h)

dimensions['mask'] = h_coarse[0].mask
dimensions['lat'] = h_coarse[1]
dimensions['lot'] = h_coarse[2]

coarse_hist['dimensions'] = dimensions

np.save('../data/probabilities.npy', coarse_hist)
