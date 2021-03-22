import numpy as np
import glob
import os
import re
import cupy as cp

from utils import timing
from cuda_correlation import cross_correlation


def csv_to_npy(filename):
    print(f"Reading {filename}...")
    array = np.loadtxt(filename, delimiter=',', )
    numpy_filename = os.path.splitext(filename)[0]
    np.save(numpy_filename, array)
    return numpy_filename


@timing
def do_correlation(m_list, l_list):
    assert(len(m_list)==len(l_list))
    r_list = [cp.empty((2*l_list[i]+1, m_list[i].shape[0], m_list[i].shape[0]), dtype=np.float32) for i in range(len(m_list))]
    cross_correlation(r_list, m_list, l_list)
    return r_list


def test_matmult(matrix_a=None, matrix_b=None):

    if not matrix_a or not matrix_b:
        rows = 63
        cols = 426000
        np.random.seed(240387)
        matrix_a = cp.random.rand(rows, cols).astype('f')
        matrix_b = cp.random.rand(cols, rows).astype('f')
        # matrix_a = np.ones((rows,cols), dtype=np.float32)
        # matrix_b = np.ones((cols, rows), dtype=np.float32)
    else:
        matrix_a = cp.array(matrix_a)
        matrix_b = cp.array(matrix_b)

    matrix_c = cp.matmul(matrix_a, matrix_b)

    return cp.asnumpy(matrix_c)


def load_data_from_file(path, lags, convert_csv=False):

    m_list = []
    l_list = []
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
            # print(f"Array {file} has shape {np.shape(array)}")
            m_list.append(array)
            l_list.append(lags[filename])
    return m_list, l_list


def test():
    DEBUG = False

    if DEBUG:
        PATH_PATTERN = "./data/TS10*"
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

    data_list = load_data_from_file(PATH_PATTERN, LAGS)
    ret = do_correlation(*data_list)
    return [cp.asnumpy(r) for r in ret]


def run(m_list, l_list):
    m_list = [np.array(i) for i in m_list]
    l_list = [int(i) for i in l_list]
    if len(l_list) != len(m_list):
        print("Data and lags have different sizes! Exiting...")
        return
    for i in range(len(m_list)):
        if l_list[i] > m_list[i].shape[1]:
            print(f"Invalid max lag value ({l_list[i]}) for filter {i}: "
                  f"setting lag value to {m_list[i].shape[1] - 1}")
            l_list[i] = m_list[i].shape[1] - 1

    ret = do_correlation(m_list, l_list)
    return [cp.asnumpy(r) for r in ret]


if __name__ == "__main__":
    res = test()
    print(res)
