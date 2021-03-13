import numpy as np
from numba import cuda
import math
# from pyculib import blas

from gpu_utils import timing


@cuda.jit('float32[:,:], float32[:,:], int32, int32')
def cuda_shift(array, array_shifted, shift, n_threads):
    # Computational blocks are unused. Each thread is assigned a single row in the matrix.
    tid = cuda.threadIdx.x
    bpg = cuda.gridDim.x

    for i in range(bpg):
        for j in range(array.shape[1]):
            if (tid + i * n_threads) >= array.shape[0]:
                continue
            if (shift > 0 and j - shift < 0) or (shift < 0 and j - shift >= array.shape[1]):
                array_shifted[tid + i * n_threads][j] = 0.0
            else:
                array_shifted[tid + i * n_threads][j] = array[tid][j - shift]


# TODO mostly pseudocode
def cross_corr_opt(matrix, matrix_shifted, lag, output):
    # corr_matrix è la correlazione tra neurone i e j per il lag corrente, corr_matrix_prev per il lag precedente
    corr_matrix = np.zeros((matrix.shape[0],matrix.shape[0]), dtype=np.float32)
    for i in range(-lag, lag+1):
        # should be called within a single block with high number of threads
        cuda_shift[1,32](matrix, matrix_shifted, i)  # conviene che matrix sia allocata come device memory, vedi https://developer.nvidia.com/blog/seven-things-numba/ punto 1
        corr_matrix_prev = np.copy(corr_matrix) # se fosse in un kernel... cudamemcopy device to device
        # pyculib.blas.Blas.gemm(matrix, matrix_shifted, )
        fast_matmul(matrix, transpose(matrix_shifted), corr_matrix) # vedi reikna, parametro alpha per normalizzare
        for j in range(corr_matrix.shape[0]):
            for k in range(corr_matrix.shape[1]):
                if abs(corr_matrix[i][j]) > abs(corr_matrix_prev[i][j]): # fabs è la funzione cublas equivalente
                    optimal_lag = -lag+index
                    output[j][1] = corr_matrix[i][j]
                    output[j][2] = index
    return output


@cuda.jit('float32[:,:,:], float32[:,:], int32')
def cuda_corr(results, data_array, lag_value):
    """
    results : numpy array 3-D: Correlation array. Shape = (neuron1, neuron2, lag)
    data_array : numpy array 2-D: Input data array. Shape = (neurons, timeseries)
    lag_value : int: base lag for the given filter.
    """
    x, y = cuda.grid(2)
    time_steps = data_array.shape[1]
    for lag in range(-lag_value, lag_value+1):
        correlation = 0.0
        if x >= results.shape[0] or y >= results.shape[1]:
            continue
        if x >= y: # using the = sign here means don't computer auto-correlation
            results[x, y, lag] = 0.0
            continue
        for t in range(time_steps):
            if t + lag < 0 or t + lag >= time_steps:
                continue
            else:
                correlation += data_array[x, t] * data_array[y, t+lag]

        correlation = correlation / (time_steps - abs(lag))
        index = lag+lag_value
        results[x, y, index] = correlation


def cross_corr(data_array, lag_array, dev_id=None):
    if dev_id:
        cuda.select_device(dev_id)

    n_filters, n_neurons, n_timestamps = data_array.shape
    max_lag = np.max(lag_array)
    results = np.ones((n_filters, n_neurons-1, n_neurons, 2 * max_lag + 1), dtype=np.float32)

    threads_per_block = (4, 4)
    bpg_x = int(math.ceil(n_neurons / threads_per_block[0]))
    bpg_y = int(math.ceil(n_neurons / threads_per_block[1]))
    blocks_per_grid = (bpg_x, bpg_y)

    for i in range(n_filters):
        cuda_corr[blocks_per_grid, threads_per_block](results[i,:,:,:], data_array[i], lag_array[i])

    return results

@timing
def run():
    # example data size is setup based on the given examples
    N_NEURONS = 4
    TIMESTEPS = 10
    N_FILTERS = 1
    TEST_LAG = 2
    np.random.seed(240387)
    data_array = np.random.rand(N_FILTERS, N_NEURONS, TIMESTEPS).astype('f')
    lag_array = np.ones(shape=(N_FILTERS), dtype=np.int32) * TEST_LAG
    results = cross_corr(data_array, lag_array)
    print(results[0][0])
    test_arr = np.copy(results[0][0])
    cuda_shift[1,32](results[0][0], test_arr, -1, 32)
    print(test_arr)


if __name__ == "__main__":
    run()