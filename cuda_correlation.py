import numpy as np
import cupy as cp

from utils import timing


def find_maximum_correlation(cp_matrix):
    maximum = cp_matrix.max(axis=0)
    minimum = cp_matrix.min(axis=0)
    max_matrix = cp.where(cp.asarray(-minimum > maximum), minimum, maximum)
    index_max_matrix = cp.abs(cp_matrix).argmax(axis=0) - cp_matrix.shape[0] // 2
    return cp.stack((max_matrix, index_max_matrix), axis=0)


def cross_correlation(result_list, data_list, lag_list):
    for i in range(len(data_list)):
        # print(f"Working on filter {i+1} of {len(data_list)}")
        tmp = cp.array(data_list[i])
        for lag in range(-lag_list[i], lag_list[i] + 1):
            # shifting a matrix using cp.pad (see numpy documentation)
            if lag > 0 :
                data_shifted = cp.pad(tmp, ((0, 0), (lag, 0)), mode='constant')[:, :-lag]
            elif lag < 0:
                data_shifted = cp.pad(tmp, ((0, 0), (0, -lag)), mode='constant')[:, -lag:]
            else:
                data_shifted = tmp
            corr_matrix = cp.matmul(tmp, cp.transpose(data_shifted)) / (tmp.shape[1] - abs(lag))
            index = lag + lag_list[i]
            result_list[i][index] = corr_matrix
        result_list[i] = find_maximum_correlation(result_list[i])


@timing
def test():
    # example data size is setup based on the given examples
    # TIMING ON RXT2080Ti: 6s with N_NEURONS = 63, TIMESTEP = 425.000, N_FILTERS = 10, LAG_LIST = 120 for each filter
    N_NEURONS = 3
    TIMESTEPS = 3
    N_FILTERS = 2
    LAG_LIST = [2] * N_FILTERS
    assert (N_FILTERS == len(LAG_LIST))

    np.random.seed(240387)
    data_list = []
    result_list = []
    lag_list = []
    for i in range(N_FILTERS):
        lag_list.append(LAG_LIST[i])
        n_lags = 2 * LAG_LIST[i] + 1
        data_list.append(np.random.random_sample((N_NEURONS, TIMESTEPS)).astype('f') * 2 - 1)
        # data_list.append(cp.ones((N_NEURONS, TIMESTEPS), dtype=np.float32))
        result_list.append(cp.empty((n_lags, N_NEURONS, N_NEURONS), dtype=np.float32))

    cross_correlation(result_list, data_list, lag_list)
    return result_list


def check_symmetric(matrix, antisym=False, rtol=1e-05, atol=1e-08):
    return np.allclose(matrix, matrix.T*(-1 if antisym else 1), rtol=rtol, atol=atol)


if __name__ == "__main__":
    res = test()