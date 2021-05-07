import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

"""
TOC:
1. read shapefiles (Lebreton or Meijer)
2. split Point objects.
3. bin the rivers into the coastal cells
4. Cluster the rivers in N groups
5. generate initial conditions for cluster
    - save the river coord inside the cluster
    - generate delayed realease randomized
    - save them, sneaky beasts
6. Compute priors.
"""


def haversine_distance_two(point_A, point_B):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lat1, lon1 = point_A
    lat2, lon2 = point_B
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def rivers_per_location(DF, loc_coords, radius, binned=False):
    """
    Input
    - DF: the pandas Dataframe with data River_sources.
    - loc_coords: tuple with the location coordinates as in (lat, lon).
    - radius: the radius in degrees around loc_coords.
    Returns
    - the dataframe around loc_coords.
    """
    if binned:
        _label = '_bin'

    else:
        _label = ''

    x_col = f'X{_label}'
    y_col = f'Y{_label}'

    X = DF[x_col]
    Y = DF[y_col]
    lat, lon = loc_coords
    mask = (X <= lon + radius) & (X > lon - radius) & \
        (Y <= lat + radius) & (Y > lat - radius)
    new_DF = DF[mask]
    return new_DF, mask


def region_filters(DF, lon_min, lon_max, lat_min, lat_max, shapefile=False):
    """
    DF is the River_sources dataframes. lat_min, lat_max, lon_min, lon_max are
    the domain limits.
    Returns the dataframe only for the region.
    """
    if shapefile:
        X = DF.geometry.x
        Y = DF.geometry.y
    else:
        X = DF['X']
        Y = DF['Y']

    mask = (X <= lon_max) & (X > lon_min) & (Y <= lat_max) & (Y > lat_min)

    new_DF = DF[mask]
    return new_DF


def nearest_coastal_cell(latidute, longitude, coord_lat, coord_lon):
    """
    Function to find the index of the closest point to a certain lon/lat value.

    - latidute and longitude are the dimensinal 1D arrays of the grid, with the
            same length.
    - coord_lat and coord_lon are the coordinates of a point.

    Returns: index (just one)
    """

    distance = np.sqrt((longitude-coord_lon)**2 + (latidute-coord_lat)**2)
    index = distance.argmin()

    return index


def convert_geopandas2pandas(geoDF):
    '''Replaces the geometry column with a X and Y columns
    There no built-in function for this in geopandas!
    '''

    L = len(geoDF)
    coord = np.zeros((L, 2))
    coord[:, 0] = geoDF.geometry.x
    coord[:, 1] = geoDF.geometry.y
    aux = pd.DataFrame(coord, columns=['X', 'Y'])
    geoDF.drop(columns=['geometry'], inplace=True)
    geoDF = pd.concat([geoDF, aux], axis=1)

    return geoDF


def rivers2coastalgrid(DF, coastal_fields):

    N = len(DF)

    coast = coastal_fields.coastal.values
    lats = coastal_fields.lat.values
    lons = coastal_fields.lon.values
    iy_coast, ix_coast = np.where(coast == 1)
    lat_coast = lats[iy_coast]
    lon_coast = lons[ix_coast]

    new_coordinates = np.zeros((N, 2))

    for i in range(N):
        x_lon = DF.iloc[i].X
        x_lat = DF.iloc[i].Y

        n_index = nearest_coastal_cell(lat_coast, lon_coast, x_lat, x_lon)
        new_coordinates[i, :] = (lon_coast[n_index], lat_coast[n_index])

    aux = pd.DataFrame(new_coordinates, columns=['X_bin', 'Y_bin'],
                       index=DF.index)
    new_DF = pd.concat([DF, aux], axis=1)

    return new_DF


# Loading caostal fields
coastal_fields = xr.load_dataset('../coastal_fields.nc')
coast = coastal_fields.coastal.values
lats = coastal_fields.lat.values
lons = coastal_fields.lon.values

X = coastal_fields.lon_mesh
Y = coastal_fields.lat_mesh

# Isolating the coastal cell coordinates
iy_coast, ix_coast = np.where(coast == 1)
lat_coast = lats[iy_coast]
lon_coast = lons[ix_coast]

South_Atlantic_region = (-70, 25, -50, -5)
