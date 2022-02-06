#--------- Useful utility functions for PCA -----------------------------#
# data:5/2/2022


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
from sklearn import preprocessing




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


def get_cons(*items, method='inter'):
    """return the intersection or union of multiple
    lists. method =inter / union
    output type: set"""

    # make them all the set types
    set_items = []
    for i, genes in enumerate(items):
        if len(genes) == len(set(genes)):
            set_items.append(set(genes))
        else:
            return ('WARNING: the {}th gene set contains duplicate genes.'.format(i + 1))

    num = len(items)
    cur_list = []
    cur_index = None
    for i, genes in enumerate(set_items):
        if cur_index == None:
            cur_list = genes;
            cur_index = i
            continue
        else:
            if method == 'inter': cur_list = cur_list.intersection(genes)
            if method == 'union': cur_list = cur_list.union(genes)
            cur_index = i
    return (cur_list)



def normalize(np_array):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np_array)
    return x_scaled

def scal_matrix(m_file_notna):
    scaler = preprocessing.StandardScaler().fit(m_file_notna)
    X_scaled = scaler.transform(m_file_notna)
    norm_m_file = pd.DataFrame(X_scaled, columns=m_file_notna.columns, index = m_file_notna.index)
    return norm_m_file


def trans_genes(m, Con_ids, gene_bankmap):
    """
    During proprocessing, take the intersection of the COSMIC list
    and mutation's gene list;
    gene_bankmap: mapping the name from COSIMIC to CCLE's genes' names.
    """
    gene_list = m.columns.map(lambda x:x.split(' ')[0])
    final_genes = get_cons(gene_list, Con_ids , method= 'inter')
    final_genes = [f"{gene} {gene_bankmap[gene]}" for gene in final_genes ]
    return m.loc[:,final_genes]
