import pandas as pd
import numpy as np
import glob
import os

DEBUG = True

if DEBUG:
    PATH_PATTERN = "data/ts10*"
else:
    PATH_PATTERN = "data/*"


def csv_to_npy(filename):
    print(f"Reading {filename}...")
    array = np.loadtxt(filename, delimiter=',',)
    numpy_filename = os.path.splitext(filename)[0]
    np.save(numpy_filename, array)


# TODO work in progress
def process_array(array):
    pass


if __name__ == "__main__":
    for file in glob.glob(PATH_PATTERN):
        file_extension = os.path.splitext(file)[1]
        if file_extension == '.csv':
            csv_to_npy(file)
        elif file_extension == '.npy':
            array = np.load(file)
            print(f"Array {file} has shape {np.shape(array)}")
            process_array(array)

