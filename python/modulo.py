import numpy as np

domain_coords = np.load("support_data.npy", allow_pickle=True)


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
