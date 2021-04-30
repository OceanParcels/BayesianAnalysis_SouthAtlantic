import numpy as np
import pandas as pd
import xarray as xr


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


def rivers_per_location(DF, loc_coords, radius):
    """
    DF is the River_sources dataframes. loc_coords is the location coordinates.
    radius is the radius of in degrees around loc_coords.
    Returns the dataframe around loc_coords.
    """
    lat, lon = loc_coords
    mask = (DF['X'] <= lon + radius) & (DF['X'] > lon - radius) \
        & (DF['Y'] <= lat + radius) & (DF['Y'] > lat - radius)
    new_DF = DF[mask]
    return new_DF


def region_filters(DF, lon_min, lon_max, lat_min, lat_max):
    """
    DF is the River_sources dataframes. lat_min, lat_max, lon_min, lon_max are
    the domain limits.
    Returns the dataframe only for the region.
    """
    mask = (DF['X'] <= lon_max) & (DF['X'] > lon_min) \
        & (DF['Y'] <= lat_max) & (DF['Y'] > lat_min)
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
