import pandas as pd; import numpy as np
import os
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings
from functools import reduce

def write_file(df, name, path):
    df.to_csv(path + name)




def split_task(total_n, n_jobs):
    """
    :param total_n:
    :param n_jobs:
    :return:
    """
    index = []
    n_col = n_jobs; n_row = total_n//n_jobs
    for i in range(n_row):
        array = np.arange(n_col) + np.array([i*n_col]*n_col)
        index.append(array)
    aug = np.array([n_row*n_col]*(total_n%n_jobs))+np.arange(total_n%n_jobs)
    index.append(aug)
    return(index)
