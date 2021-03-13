from numba import cuda, float32
import time

N_THREADS = 16


def print_debug_info():
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))


def timing(func):
    def wrapper(*args):
        if "kernel" in func.__name__:
            start = time.time()
            func(*args)
            end = time.time()
            print(f"Kernel took {end - start} seconds (including compilation)")
            start = time.time()
            func(*args)
            end = time.time()
            print(f"Kernel took {end - start} seconds (excluding compilation)")
        else:
            start = time.time()
            func(*args)
            end = time.time()
            print(f"Host function took {end - start} seconds")
    return wrapper


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
@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)
    sB = cuda.shared.array(shape=(N_THREADS, N_THREADS), dtype=float32)

    x, y = cuda.grid(2)

    # thread indexes
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # blocks per grid
    bpg_x = cuda.gridDim.x
    bpg_y = cuda.gridDim.y

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of N_THREADS-long vectors.
    tmp = 0.
    for i in range(max(bpg_x, bpg_y)):
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