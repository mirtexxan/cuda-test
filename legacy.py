from numba import cuda, float32
import math
import cupy as cp

from utils import timing


@timing
def run_numba_kernel(func, bpg, tpb, *args):
    func[bpg, tpb](*args)


# for testing purposes
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def naive_matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


# for testing purposes (NOTE: is very slow because it does not avoid useless computations)
N_THREADS = 16
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def fast_matmul(A, B, C):

    x, y = cuda.grid(2)
    # thread indexes
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)
    sB = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of N_THREADS-long vectors.
    tmp = 0.
    for i in range(N_THREADS):
        # Preload data into shared memory
        sA[tx, ty] = 0.
        sB[tx, ty] = 0.
        if x < A.shape[0] and (ty + i * N_THREADS) < A.shape[1]:
            sA[tx, ty] = A[x, ty + i * N_THREADS]
        if y < B.shape[1] and (tx + i * N_THREADS) < B.shape[0]:
            sB[tx, ty] = B[tx + i * N_THREADS, y]
        # Wait until all threads finish preloading
        cuda.syncthreads()
        # Computes partial product on the shared memory
        for j in range(N_THREADS):
            tmp += sA[tx, j] * sB[j, ty]
        # Wait until all threads finish computing
        cuda.syncthreads()
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = tmp


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


@cuda.jit('float32[:,:,:], float32[:,:], int32')
def cuda_corr_naive(result, data_array, lag_value):
    """
    results : numpy array 3-D: Correlation array. Shape = (lag, neuron1, neuron2)
    data_array : numpy array 2-D: Input data array. Shape = (neurons, timeseries)
    lag_value : int: base lag for the given filter.
    """
    x, y = cuda.grid(2)
    time_steps = data_array.shape[1]
    for lag in range(-lag_value, lag_value + 1):
        correlation = 0.0
        if x >= result.shape[1] or y >= result.shape[2]:
            continue
        if x >= y:  # using the = sign here means don't computer auto-correlation
            result[lag + lag_value, x, y] = 0.0
            continue
        for t in range(time_steps):
            if t + lag < 0 or t + lag >= time_steps:
                continue
            else:
                correlation += data_array[x, t] * data_array[y, t + lag]

        correlation = correlation / (time_steps - abs(lag))
        result[lag + lag_value, x, y] = correlation


def cross_corr_naive(result_list, data_list, lag_list):
    if len(data_list) != len(data_list) or len(data_list) != len(result_list):
        print("Mismatched data and lags input")
        return

    threads_per_block = (4, 8)
    for i in range(len(data_list)):
        bpg_x = int(math.ceil(data_list[i].shape[0] / threads_per_block[0]))
        bpg_y = int(math.ceil(data_list[i].shape[1] / threads_per_block[1]))
        blocks_per_grid = (bpg_x, bpg_y)
        cuda_corr_naive[blocks_per_grid, threads_per_block](result_list[i], data_list[i], lag_list[i])

    for i in enumerate(result_list):
        result_list[i] = find_maximum_correlation_opt(result_list[i])


# there is no significant improvement while the input matrix stays small (and is quite unreadable)
def find_maximum_correlation_opt(cp_matrix):
    index_max_matrix = cp.abs(cp_matrix).argmax(axis=0)
    mesh = cp.meshgrid(tuple(cp.arange(cp_matrix.shape[-1])), tuple(cp.arange(cp_matrix.shape[-2])))
    max_matrix = cp_matrix[index_max_matrix.ravel(), mesh[1].ravel(), mesh[0].ravel()].reshape(cp_matrix.shape[-2], cp_matrix.shape[-1])
    return max_matrix, index_max_matrix
