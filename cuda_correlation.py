import numpy as np
from numba import cuda
import math


@cuda.jit()
def cuda_corr(res_c, dft, ml, max_lag, n):
    """
    Cuda kernel computing the correlation vlaues for pair correlation analysis.

    Parameters
    ----------
    res_c : numpy array 2-D
        Result array.
    dft : numpy array 2-D
        Data array.
    ml : numpy array 1-D
        Array containig all lags.
    m_lag : int
        Maximal lag.
    n : int
        Number of units.

    Returns
    -------
    None.

    """
    ts, x, y = cuda.grid(3)
    if ts >= res_c.shape[0] or x >= res_c.shape[1] or y >= res_c.shape[2]:
        return
    if x >= y:
        return
    mll = ml[ts]
    for lag in range(-mll, mll+1):
        tmp = 0.0
        for t in range(dft.shape[2]):
            if t + lag < 0 or t + lag >= dft.shape[2]:
                continue
            else:
                tmp = tmp + dft[ts, x, t + lag] * dft[ts, y, t] # + - change
        tmp = tmp / (dft.shape[2] - abs(lag))
        idx = lag + max_lag
        res_c[ts, x, y, idx] = tmp
    return


def cross_corr(array, bin_lags, m_lag, n, dev_id=None):
    """
    Umbrella function for cuda based correlation analysis on pairwise level.
    Calculates the cuda-gird and parses every necessary array to GPU.

    Parameters
    ----------
    array : numpy array 3-D
        Moving average filtered data.
    bin_lags : numpy aaray 1-D
        diffrent lags for diffrent timescales
    m_lag : int
        maximal lag for all timescales.
    n : int 
        number of unit
    dev_id : int
        device id (for multiple gpus)

    Returns
    -------
    lin_corr_mat : numpy array 3-D
        3-D array of best correlation values.
    L : numpy array 3-D
        3-D array of correspon.
    interaction : numyp array 3-D
        DESCRIPTION.

    """
    
    if dev_id:
        cuda.select_device(dev_id)
    THREADS_PER_BLOCK = (4, 4, 4)
    ts, n, T = array.shape
    res = np.zeros((ts, n, n, 2*m_lag+1))

    blockspergrid_x = int(math.ceil(ts / THREADS_PER_BLOCK[0]))
    blockspergrid_y = int(math.ceil(n / THREADS_PER_BLOCK[1]))
    blockspergrid_z = int(math.ceil(n / THREADS_PER_BLOCK[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_corr[blockspergrid, THREADS_PER_BLOCK](res, array, bin_lags, m_lag, n)
    
    return res
