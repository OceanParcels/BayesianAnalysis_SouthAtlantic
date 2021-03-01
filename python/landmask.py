import os
from copy import deepcopy
import progressbar
import xarray
from numpy import array
from parcels import FieldSet
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


def get_coastal_cells(landmask):
    # Going through ocean cells to see which are next to land
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype('int')

    return coastal


def get_shore_cells(landmask):
    # Going through land cells to see which are next to the ocean
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')

    return shore


def create_border_current(landmask):

    coastal = get_coastal_cells(landmask)
    Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
    Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)

    v_x = np.ma.masked_where(coastal == 0, -Lx)
    v_y = np.ma.masked_where(coastal == 0, -Ly)

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal cells between land create a problem. Magnitude there is zero
    # I force it to be one to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x/magnitude
    v_y = v_y/magnitude
    v_x.set_fill_value(value=0)
    v_y.set_fill_value(value=0)

    return v_x.data, v_y.data


# getting my data saved for simulations

file_path = "../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc"
indices = {'lat': range(1, 900), 'lon': range(1284, 2460)}

land_mask = make_landmask(file_path, indices)
coastal_cells = get_coastal_cells(land_mask)
coastal_u, coastal_v = create_border_current(land_mask)
np.save('../landmask.npy', land_mask)
np.save('../coastal_cells.npy', coastal_cells)
np.save('../coastal_u.npy, ', coastal_u)
np.save('../coastal_v.npy, ', coastal_v)
