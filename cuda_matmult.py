from numba import cuda, float32
import numpy as np
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
N_THREADS = 16


@cuda.jit()
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)
    sB = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of N_THREADS-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * N_THREADS]
        sB[tx, ty] = B[tx + i * N_THREADS, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(N_THREADS):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


if __name__ == "__main__":

    if not cuda.is_available():
        print("Cuda is not available on this machine.")
        exit()

    MATRIX_SIZE = 1000
    MATRIX_SHAPE = (MATRIX_SIZE, MATRIX_SIZE)

    A = np.ones(MATRIX_SHAPE)
    B = np.ones(MATRIX_SHAPE) * 2
    C = np.ndarray(MATRIX_SHAPE)

    threadsperblock = (N_THREADS, N_THREADS)
    n_blocks = int(math.ceil(MATRIX_SIZE / N_THREADS))
    blockspergrid = (n_blocks, n_blocks)

    fast_matmul[blockspergrid, threadsperblock](A, B, C)
    print(C)



