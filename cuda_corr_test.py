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
    for l in range(-mll, mll+1):
        tmp = 0.0
        for t in range(dft.shape[2]):
            if t + l < 0 or t + l >= dft.shape[2]:
                continue
            else:
                tmp = tmp + dft[ts, x, t + l] * dft[ts, y, t] # + - change
        tmp = tmp / (dft.shape[2] - abs(l))
        idx = l + max_lag  
        res_c[ts, x, y, idx] = tmp
    return

def cross_corr(dft_c, bin_lags, m_lag, n, dev_id=None):
    """
    Umbrella function for cuda based correlation analysis on pairwise level.
    Calculates the cuda-gird and parses every necessary array to GPU.
    

    Parameters
    ----------
    dft_c : numpy array 3-D
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
    
    if dev_id is not None:
        cuda.select_device(dev_id)
    tpb = 4
    ts, n, T = dft_c.shape
    res = np.zeros((ts, n, n, 2*m_lag+1))
    
    
    threadsperblock = (tpb, tpb, tpb)
    blockspergrid_x = int(math.ceil(ts / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(n / threadsperblock[1]))
    blockspergrid_z = int(math.ceil(n / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    cuda_corr[blockspergrid, threadsperblock](res, dft_c, bin_lags, m_lag, n)
    
    return res
