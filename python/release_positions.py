"""
This script creates the clusters, the priors and the release points or initial
conditions for the experimente SAG_experiment.py.

It uses the Meijer2021_midpoint_emissions GIS dataset.

It looks kinda long because the docstrings. Go straight to line 264.

**I would like to generalize the script to also do this with Lebretons2018
dataset.
"""
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


def haversine_distance_two(point_A, point_B):
    """Calculates the great circle distance between two points
    on the Earth.

    Parameters
    ----------
    point_A: tuple
        containing the (latitude, longitude) in decimal degrees coordinates of
        point A.
    point_B: tuple
        containing the (latitude, longitude) in decimal degrees coordinates of
        point B.

    Returns
    -------
    km: float
        the distance in km between point A and point B
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


def region_filters(DF, lon_min, lon_max, lat_min, lat_max, shapefile=False):
    """Takes a Dataframe with the all the rivers information and filters the data
    from the rivers in an especific region.

    Parameters
    ----------
    DF: Dataframe
        is the River_sources dataframe.
    lat_min, lat_max, lon_min, lon_max: float, float, float float
        domain limits.
    shapefile: bool, optional
        True when dealing with a geopandas Dataframe.

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe rivers in the especific region.
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
    """Function to find the index of the closest point to a certain lon/lat value.

    Parameters
    ----------
    latidute: 1D array
        the latitude 1D array of the grid.
    longitude: 1D array
        the longitude 1D array of the grid.
    coord_lat: float
        latitudinal coordinate of a point.
    coord_lon: float
        longitudinal coordinate of a point.

    Returns
    -------
    index: int array
        The index of the cell from the latidute and longitude arrays. 1 index
        for both arrays.
    """

    distance = np.sqrt((longitude-coord_lon)**2 + (latidute-coord_lat)**2)
    index = distance.argmin()

    return index


def convert_geopandas2pandas(geoDF):
    '''Replaces the geometry column with a X and Y columns
    There no built-in function for this in geopandas!

    Parameters
    ----------
    geoDF: Dataframe
        a GeoPandas Dataframe with geometry column.

    Returns
    -------
    geoDF: Dataframe
        a pandas Dataframe with X, Y coordinates for each point.
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
    """Takes the river locations in the riverine plastic discharge dataset and
    it bins it into the coastal_cells of the model to be used.

    Parameters
    ----------
    DF: Dataframe
        Pandas dataframe with the riverine plastic discharge and locations.
    coastal_fields: xarray Dataset
        must be generated with landmask.py and contains all the coastal fields
        of the velocity fields to be used. e.g. SMOC dataset.

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe with the binned rivers into the coastal_cells.
    """
    N = len(DF)
    coast = coastal_fields.coast.values
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

    counts = new_DF.groupby(['X_bin', 'Y_bin']).count().loc[:, 'X'].values
    new_DF = new_DF.groupby(['X_bin', 'Y_bin']).sum()
    new_DF['merged_rivers'] = counts
    new_DF.reset_index(inplace=True)
    new_DF.drop(labels=['X', 'Y'], axis=1, inplace=True)

    return new_DF


def center_of_mass(DF):
    """Computes the center of mass from a river dataframe.
    Warning: Only works with Meijer dataset.

    Parameters
    ----------
    DF: Dataframe
        The file location of the spreadsheet
    coastal_fields : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    new_DF: Dataframe
        a pandas Dataframe with the binned rivers into the coastal_cells.

    """
    x = DF.X_bin
    y = DF.Y_bin
    m = DF.dots_exten  # this is so annoying only for Meijer.
    M = m.sum()
    return sum(m*y)/M, sum(m*x)/M


def rivers_per_location(DF, loc_coords, radius, binned=False, tolerance=0.1):
    """It cluster the rivers in a square with sides 2*radius. The clustering is
    done iteratively using the center of mass.

    Parameters
    ----------
    DF: Dataframe
        the pandas Dataframe with data River_sources.
    loc_coords: tuple
        containing the location coordinates as in (lat, lon).
    radius: float
        the radius in degrees around loc_coords.
    binned: bool, optional
        default to False. True if the Dataframe is binned using
        rivers2coastalgrid.
    tolerance: float, optional
        the tolerance in km to stop the iterations.

    Returns
    -------
    mask: list
        contains the index with the rivers around loc_coords within radius.
    CM: tuple
        a tuple with the (lat, lon) coordinates of the center of mass.
    """
    if binned:
        _label = '_bin'

    else:
        _label = ''

    x_col = f'X{_label}'
    y_col = f'Y{_label}'

    lat, lon = loc_coords
    mask = (DF[x_col] <= lon + radius) & (DF[x_col] > lon - radius) & \
        (DF[y_col] <= lat + radius) & (DF[y_col] > lat - radius)
    CM = center_of_mass(DF[mask])
    dist = haversine_distance_two((lat, lon), CM)

    while dist > tolerance:
        lat, lon = CM
        mask = (DF[x_col] <= lon + radius) & (DF[x_col] > lon - radius) & \
            (DF[y_col] <= lat + radius) & (DF[y_col] > lat - radius)
        CM = center_of_mass(DF[mask])
        dist = haversine_distance_two((lat, lon), CM)

    loc_df = DF[mask]
    p = pd.DataFrame({'p': loc_df['dots_exten']/loc_df['dots_exten'].sum()})
    loc_df = loc_df.drop(['dots_exten'], axis=1)
    loc_df = pd.concat([loc_df, p], axis=1)
    loc_df.reset_index(inplace=True)

    return mask, CM


###############################################################################
# Parameters
###############################################################################
r = 1  # radius for clusters.
N = 100000  # Number of particles realesed per source.
South_Atlantic_region = (-70, 25, -50, -5)  # the region to study
save_priors = True  # True for saving the priors.

###############################################################################
# Load all requiered data
###############################################################################

# the coastal fields dataset.
coastal_fields = xr.load_dataset('../coastal_fields.nc')
coast = coastal_fields.coast.values
lats = coastal_fields.lat.values
lons = coastal_fields.lon.values
X = coastal_fields.lon_mesh
Y = coastal_fields.lat_mesh

# reshape the coastal fields in 1D arrays
iy_coast, ix_coast = np.where(coast == 1)
lat_coast = lats[iy_coast]
lon_coast = lons[ix_coast]


# Read the GIS Shapefile from Meijer
path = '../data/sources/Meijer2021_midpoint_emissions/'
river_discharge = convert_geopandas2pandas(gpd.read_file(path))
river_discharge = region_filters(river_discharge, *South_Atlantic_region)
river_discharge = rivers2coastalgrid(river_discharge, coastal_fields)

# compute total discharged plastic in South Atlantic
total_plastic = river_discharge['dots_exten'].sum()

# sort the rivers by discharge from large to small.
river_discharge = river_discharge.sort_values(['dots_exten'], ascending=False)
river_discharge.reset_index(inplace=True, drop=True)

# define the cluster river locations (by eye)
cluster_locations = {'Congo': (-5.6442, 12.1375),
                     'Cape-Town': (-33.93, 18.56),
                     'Rio-de-la-Plata': (-33.9375, -58.5208),
                     'Porto-Alegre': (-30.051, -51.285),
                     'Santos': (-23.9875, -46.2958),
                     'Paraiba': (-21.6208, -41.0375),
                     'Itajai': (-26.9125, -48.6458),
                     'Rio-de-Janeiro': (-23.01250, -43.32083),
                     'Salvador': (-13.017065, -38.579832),
                     'Recife': (-8.09, -34.88)}

# Move them into the coastal cells. Maybe this is not necessary.
grid_cluster_centers = {}
for loc in cluster_locations:

    indx = nearest_coastal_cell(lat_coast, lon_coast, *cluster_locations[loc])
    grid_cluster_centers[loc] = (lat_coast[indx], lon_coast[indx])

###############################################################################
# Generate the Clusters, release points and priors
###############################################################################
release_points = {}
priors = {}

cluster_percent = 0  # counter for the percentege of plastic in the clusters.
merged_rivers = 0  # counter for the number or rivers in all the clusters.

for i, loc in enumerate(cluster_locations):
    print(loc)
    # get the local DF for cluster
    mask, _CM = rivers_per_location(river_discharge, cluster_locations[loc],
                                    r, binned=True)
    loc_df = river_discharge[mask]

    # number of rivers merged
    numer_rivers = loc_df['merged_rivers'].sum()
    merged_rivers += numer_rivers

    # compute prior for cluster
    loc_percent = loc_df['dots_exten'].sum()/total_plastic
    priors[loc] = [loc_percent, numer_rivers]
    cluster_percent += loc_df['dots_exten'].sum()/total_plastic

    # compute the weights for each river within the cluster
    p = pd.DataFrame({'p': loc_df['dots_exten']/loc_df['dots_exten'].sum()})

    loc_df = loc_df.drop(['dots_exten'], axis=1)  # droppin this, dont need it
    loc_df = pd.concat([loc_df, p], axis=1)
    loc_df.reset_index(drop=True, inplace=True)

    # IMPORTANT STEP. Samples randomly the locations of the rivers N times
    # according to the weigths 'p'. This creates the initital conditions for
    # the experiment.
    release_points[loc] = loc_df.sample(n=N, replace=True, weights='p')

priors = pd.DataFrame(priors).T
priors = priors.rename(columns={0: 'Mean', 1: 'merged_rivers'})
priors['Mean'] = priors['Mean']/priors['Mean'].sum()  # nomarlizing

###############################################################################
# Save the stuff
###############################################################################
np.save('../river_sources.npy', cluster_locations, allow_pickle=True)
np.save('../release_positions.npy', release_points, allow_pickle=True)
if save_priors:
    priors.to_csv('../data/analysis/priors_river_inputs.csv')
