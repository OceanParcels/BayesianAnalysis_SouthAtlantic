from netCDF4 import Dataset
import numpy as np
import xarray as xr


def make_landmask(path2output, indices):
    """Returns landmask where land = 1 and ocean = 0.

    - path2output is the path to an output of a model (netcdf file expected).
    - indices is a dictionary such as:
        indices = {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.

    Output: 2D array containing the landmask. Were the landcells are 1 and
            the ocean cells are 0.

    Warning: tested for the CMEMS model outputs where I asume that the variable
            uo exists.
    """
    datafile = Dataset(path2output)
    landmask = datafile.variables['uo'][0, 0, indices['lat'], indices['lon']]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype('int')

    return landmask


def get_coastal_cells(landmask):
    """Function that detects the coastal cells, i.e. the ocean cells directly
    next to land. Computes the Laplacian of landmask.

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
    next to the ocean. Computes the Laplacian of landmask.

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


def create_border_current(landmask, double_cell=False):
    """Function that creates a border current 1 m/s away from the shore.

    - landmask: the land mask built using `make_landmask`.
    - double_cell: Boolean for determining if you want a double cell.
      Default set to False.

    Output: two 2D arrays, one for each camponent of the velocity.
    """
    shore = get_shore_cells(landmask)
    coastal = get_coastal_cells(landmask)
    Ly = np.roll(landmask, -1, axis=0) - np.roll(landmask, 1, axis=0)
    Lx = np.roll(landmask, -1, axis=1) - np.roll(landmask, 1, axis=1)

    if double_cell:
        v_x = -Lx*(coastal+shore)
        v_y = -Ly*(coastal+shore)
    else:
        v_x = -Lx*(coastal)
        v_y = -Ly*(coastal)

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal cells between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x/magnitude
    v_y = v_y/magnitude

    return v_x, v_y


def distance_to_shore(landmask, dx=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_cells` algorithm.

    - landmask: the land mask built using `make_landmask` function.
    - dx: the grid cell dimesion. This is a crude approximation of the real
    distance (be careful).

    Output: 2D array containing the distances from shore.
    """
    ci = get_coastal_cells(landmask)
    landmask_i = landmask + ci
    dist = ci
    i = 0

    while i < dist.max():
        ci = get_coastal_cells(landmask_i)
        landmask_i += ci
        dist += ci*(i+2)
        i += 1

    return dist*dx


def generate_dataset(path2output, indices, output_path):
    """Creates a netCDF file with all the fields needed to run
    SAG_experiment.py.

    - path2output is the path to an output of a model (netcdf file expected).
    - indices is a dictionary such as:
        indices = {'lat': slice(1, 900), 'lon': slice(1284, 2460)}.

    - output_path is the output path and name of the netCDF file.

    Output: xarray dataset.
    """
    model = xr.load_dataset(path2output)
    lons = model['longitude'][indices['lon']]  # * unpacks the tuple
    lats = model['latitude'][indices['lat']]
    X, Y = np.meshgrid(lons, lats)

    landmask = make_landmask(path2output, indices)
    coastal_cells = get_coastal_cells(landmask)
    shore_cells = get_shore_cells(landmask)
    coastal_u, coastal_v = create_border_current(landmask)
    distance2shore = distance_to_shore(landmask, dx=9.26)  # km

    ds = xr.Dataset(
        data_vars=dict(
            landmask=(["y", "x"], landmask),
            coastal=(["y", "x"], coastal_cells),
            shore=(["y", "x"], shore_cells),
            coastal_u=(["y", "x"], coastal_u),
            coastal_v=(["y", "x"], coastal_v),
            distance2shore=(["y", "x"], distance2shore),
            lat_mesh=(["y", "x"], Y),
            lon_mesh=(["y", "x"], X),),

        coords=dict(lon=(["x"], lons.values),
                    lat=(["y"], lats.values),),

        attrs=dict(description="setup files for SAG_experiment.py.",
                   index_lat=(indices['lat'].start, indices['lat'].stop),
                   index_lon=(indices['lon'].start, indices['lon'].stop)))

    ds.to_netcdf(output_path)


# Getting my data saved for simulations
print('Generating setup_fields.nc')

file_path = "../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc"
indices = {'lat': range(1, 960), 'lon': range(1284, 2460)}
outfile = '../coastal_fields.nc'

generate_dataset(file_path, indices, outfile)
