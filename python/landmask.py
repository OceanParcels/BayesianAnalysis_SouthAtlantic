from netCDF4 import Dataset
import numpy as np


def make_landmask(fielddata, indices):
    """Returns landmask where land = 1 and ocean = 0
    fielddata is a netcdf file.
    indices is a dictionary such as:
    indices = {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.
    """
    datafile = Dataset(fielddata)

    landmask = datafile.variables['uo'][0, 0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')

    return landmask


file_path = "../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc"
indices = {'lat': range(1, 900), 'lon': range(1284, 2460)}

land_mask = make_landmask(file_path, indices)
np.save('../landmask.npy', land_mask)
