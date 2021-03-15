from numba import cuda
import numpy as np
import glob
import os
import re

from gpu_utils import run_numba_kernel, fast_matmul, naive_matmul
from cuda_correlation import cross_corr_naive, cross_corr_opt

DEBUG = True

if DEBUG:
    PATH_PATTERN = "./data/TS1*"
else:
    PATH_PATTERN = "./data/*"

LAGS = {"TS01": 20,
        "TS02": 20,
        "TS03": 41,
        "TS04": 60,
        "TS05": 100,
        "TS06": 120,
        "TS07": 80,
        "TS08": 50,
        "TS09": 34,
        "TS10": 25
        }


def csv_to_npy(filename):
    print(f"Reading {filename}...")
    array = np.loadtxt(filename, delimiter=',', )
    numpy_filename = os.path.splitext(filename)[0]
    np.save(numpy_filename, array)
    return numpy_filename


# TODO work in progress: here we should call correlation functions from cuda_correlation.py
def process_list(list):
    n_filters = len(list)
    print(n_filters)



def test_matmult(matrix_a=None, matrix_b=None):
    if not cuda.is_available():
        print("Cuda is not available on this machine.")
        exit()

    if not matrix_a or not matrix_b:
        rows = 63
        cols = 426000
        matrix_a = np.random.rand(rows, cols).astype('f')
        matrix_b = np.random.rand(cols, rows).astype('f')
        # np.random.seed(240387)
        # matrix_a = np.ones((rows,cols), dtype=np.float32)
        # matrix_b = np.ones((cols, rows), dtype=np.float32)
    else:
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)

    matrix_c = np.empty((matrix_a.shape[0], matrix_b.shape[1]), dtype=np.float32)

    args = (matrix_a, matrix_b, matrix_c)
    n_threads = 16
    tpb = (n_threads, n_threads)
    n_blocks_x = (matrix_a.shape[0] + n_threads - 1) // n_threads
    # n_blocks_y = (matrix_a.shape[1] + n_threads - 1) // n_threads
    bpg = (n_blocks_x, n_blocks_x)
    print(f"Running matrix multiplication with bpg={bpg} and tpb={tpb}")
    run_numba_kernel(naive_matmul, bpg, tpb, *args)
    return matrix_c


def load_data(path, convert_csv=False):
    matrix_lag_list = []
    if convert_csv:
        for file in glob.glob(path):
            file_extension = os.path.splitext(file)[1]
            if file_extension == '.csv':
                csv_to_npy(file)
    for file in glob.glob(path):
        file_extension = os.path.splitext(file)[1]
        if file_extension == '.npy':
            array = np.load(file).astype('f')
            # set any float <1e-15 to 0 (for efficiency)
            np.around(array, decimals=15, out=array)
            def subfunc(m):
                return m.group(1)+m.group(2) if len(m.group(2)) == 2 else m.group(1)+"0"+m.group(2)
            filename = re.sub(r'[a-z./]*([A-Za-z]+)([0-9]+)\.[a-z]+.', subfunc, file)
            if DEBUG:
                print(f"Array {file} has shape {np.shape(array)}")
            matrix_lag_list.append((filename, array, LAGS[filename]))
    return sorted(matrix_lag_list)


if __name__ == "__main__":
    matrix_lag_list = load_data(PATH_PATTERN)
    process_list(matrix_lag_list)
