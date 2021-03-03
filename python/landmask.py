from netCDF4 import Dataset
import numpy as np


def make_landmask(fielddata, indices):
    """Returns landmask where land = 1 and ocean = 0
    - fielddata is the path to an output of a model (netcdf file expected).
    - indices is a dictionary such as:
        indices = {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.

    Output: 2D array containing the landmask. Were the landcells are 1 and
            the ocean cells are 0.

    Warning: tested for the CMEMS model outputs where I asume that the variable
            uo exists.
    """
    datafile = Dataset(fielddata)
    landmask = datafile.variables['uo'][0, 0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')

    return landmask


def make_grid(fielddata, indices):
    """Returns landmask where land = 1 and ocean = 0
    - fielddata is the path to an output of a model (netcdf file expected).
    - indices is a dictionary such as:
        indices = {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.

    Output: 2D masked array array containing the landmask, 1D array with the
            latitudes and a 1D array with the longitudes.

    Warning: tested for the CMEMS model outputs where I asume that the variable
            uo exists.
    """
    datafile = Dataset(fielddata)
    landmask = datafile.variables['uo'][0, 0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    lat = datafile.variables['latitude'][:].data
    lon = datafile.variables['longitude'][:].data

    return landmask, lat, lon


def get_coastal_cells(landmask):
    """Function that detects the coastal cells, i.e. the ocean cells directly
    next to land.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the coastal cells, the coastal cells are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype('int')

    return coastal


def get_shore_cells(landmask):
    """Function that detects the shore cells, i.e. the land cells directly
    next to the ocean.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore cells, the shore cells are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4*landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype('int')

    return shore


def create_border_current(landmask):
    """Function that creates a border current 1 m/s away from the shore.txt
    - landmask: the land mask built using `make_landmask`.

    Output: two 2D arrays, one for each camponent of the velocity.
    """
    coastal = get_coastal_cells(landmask)
    Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
    Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)

    v_x = np.ma.masked_where(coastal == 0, -Lx)
    v_y = np.ma.masked_where(coastal == 0, -Ly)

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal cells between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x/magnitude
    v_y = v_y/magnitude
    v_x.set_fill_value(value=0)
    v_y.set_fill_value(value=0)

    return v_x.data, v_y.data


def distance_to_shore(landmask, iterations=20, dx=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_cells` algorithm.
    - landmask: the land mask built using `make_landmask` function.
    - iterations: the number of cells to iterate from shore. By default is set
    to 20, so you will get the distance from shore only for 20 cells from shore
    - dx: the grid cell dimesion. This is a crude approximation of the real
    distance (be careful).

    Output: 2D array containing the distances from shore.
    """
    ci = get_coastal_cells(landmask)
    landmask_i = landmask + ci
    dist = ci

    for i in range(iterations):
        ci = get_coastal_cells(landmask_i)
        landmask_i += ci
        dist += ci*(i+1)
    return dist*dx


# Getting my data saved for simulations

file_path = "../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc"
indices = {'lat': range(1, 900), 'lon': range(1284, 2460)}

land_mask = make_landmask(file_path, indices)
coastal_cells = get_coastal_cells(land_mask)
coastal_u, coastal_v = create_border_current(land_mask)
np.save('../landmask.npy', land_mask)
np.save('../coastal_cells.npy', coastal_cells)
np.save('../coastal_u.npy, ', coastal_u)
np.save('../coastal_v.npy, ', coastal_v)
