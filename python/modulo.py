import numpy as np

# domain_coords = np.load("support_data.npy", allow_pickle=True)


def coarsen(array, x, y, factor):
    """
    Function to coarsen a 2d array by a especified factor.
    WARNING: If array dimensions are not multiples of the
    factor it trims the array to make it a multiple of the
    factor.
    array: 2d numpy array.
    factor: positive integer.

    Returns: 2d numpy array.

    warning: if the shape of array is not a multiple of factor
    You loose %factor number of rows/colums.
    """
    q = array.shape[0]//factor
    w = array.shape[1]//factor
    k = array.shape[0] - array.shape[0] % factor
    m = array.shape[1] - array.shape[1] % factor
    array = array[:k, :m]
    aux = array.reshape((array.shape[0]//factor, factor,
                         array.shape[1]//factor, factor))
    array = np.sum(aux, axis=(1, 3))
    x = np.linspace(x[0], x[-1], q)
    y = np.linspace(y[0], y[-1], w)
    return array, x, y


def coarsen_1D(array, x, factor):
    """
    Function to coarsen a 1D array by a especified factor.
    If the array dimensions are not multiples of the
    factor it adds the missing cells to make the length of the array divisible
    by the factor.
    array: 1D numpy array.
    x: 1D array with the dimension of `array`
    factor: positive integer.

    Returns: 1D numpy coarse array, updated x dimensions.
    """
    fill = factor - array.shape[0] % factor
    k = array.shape[0] + fill

    array = np.append(array[:k], np.zeros(fill))

    aux = array.reshape((array.shape[0]//factor, factor))
    array = np.sum(aux, axis=1)

    dx = np.diff(x)[0]
    x = np.arange(x[0], x[-1] + dx*fill, dx*factor)

    return array, x


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
