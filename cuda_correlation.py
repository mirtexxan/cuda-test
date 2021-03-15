import numpy as np
from numba import cuda
import math
import cupy as cp

from gpu_utils import timing


# TODO: implement maximum and fix how data is returned to the caller
def cross_corr_opt(result_list, data_list, lag_list):
    for i in range(len(data_list)):
        for lag in range(-lag_list[i], lag_list[i] + 1):
            # shifting a matrix using cp.pad (see numpy documentation)
            if lag > 0:
                data_shifted = cp.pad(data_list[i], ((0,0),(lag,0)), mode='constant')[:,:-lag]
            elif lag < 0:
                data_shifted = cp.pad(data_list[i], ((0, 0), (0, -lag)), mode='constant')[:, -lag:]
            else:
                data_shifted = data_list[i]
            corr_matrix = cp.matmul(data_list[i], cp.transpose(data_shifted)) / (data_list[i].shape[1]-abs(lag))
            index = lag + lag_list[i]
            result_list[i][index] = corr_matrix
        # compute the maximum correlation
        print(result_list[i])
        print("-----------------------")
        print(np.maximum.reduce(cp.asnumpy(cp.abs(result_list[i]))))


@cuda.jit('float32[:,:,:], float32[:,:], int32')
def cuda_corr_naive(result, data_array, lag_value):
    """
    results : numpy array 3-D: Correlation array. Shape = (lag, neuron1, neuron2)
    data_array : numpy array 2-D: Input data array. Shape = (neurons, timeseries)
    lag_value : int: base lag for the given filter.
    """
    x, y = cuda.grid(2)
    time_steps = data_array.shape[1]
    for lag in range(-lag_value, lag_value+1):
        correlation = 0.0
        if x >= result.shape[1] or y >= result.shape[2]:
            continue
        if x >= y: # using the = sign here means don't computer auto-correlation
            result[lag + lag_value, x, y] = 0.0
            continue
        for t in range(time_steps):
            if t + lag < 0 or t + lag >= time_steps:
                continue
            else:
                correlation += data_array[x, t] * data_array[y, t+lag]

        correlation = correlation / (time_steps - abs(lag))
        result[lag + lag_value, x, y] = correlation


def cross_corr_naive(result_list, data_list, lag_list):
    if len(data_list) != len(data_list) or  len(data_list) != len(result_list):
        print("Mismatched data and lags input")
        return

    threads_per_block = (4, 8)
    for i in range(len(data_list)):
        bpg_x = int(math.ceil(data_list[i].shape[0] / threads_per_block[0]))
        bpg_y = int(math.ceil(data_list[i].shape[1] / threads_per_block[1]))
        blocks_per_grid = (bpg_x, bpg_y)
        # print(f"Launching kernel {i+1} of {len(data_list)}")
        cuda_corr_naive[blocks_per_grid, threads_per_block](result_list[i], data_list[i], lag_list[i])


@timing
def test(opt=True):
    # example data size is setup based on the given examples
    # WORST CASE SCENARIO: 174s with N_NEURONS = 63, TIMESTEP = 425.000, N_FILTERS = 10, TEST_LAG = 120 for each filter
    # WORST CASE SCENARIO (OPT): 6s with N_NEURONS = 63, TIMESTEP = 425.000, N_FILTERS = 10, TEST_LAG = 120 for each filter
    N_NEURONS = 3
    TIMESTEPS = 10
    N_FILTERS = 1
    TEST_LAG = [2]*N_FILTERS
    assert(N_FILTERS==len(TEST_LAG))

    np.random.seed(240387)
    data_list = []
    result_list = []
    lag_list = []
    for i in range(N_FILTERS):
        lag_list.append(TEST_LAG[i])
        n_lags = 2 * TEST_LAG[i] + 1
        data_list.append(cp.random.rand(N_NEURONS, TIMESTEPS).astype('f'))
        # data_list.append(cp.ones((N_NEURONS, TIMESTEPS), dtype=np.float32))
        result_list.append(cp.empty((n_lags, N_NEURONS, N_NEURONS), dtype=np.float32))

    if opt:
        cross_corr_opt(result_list, data_list, lag_list)
        # print(result_list[0][:])
    else:
        cross_corr_naive(result_list, data_list, lag_list)
        # print(result_list[0][:])
    return result_list


if __name__ == "__main__":
    res = test()
    # TODO: fix res is always None here
    print(res)
