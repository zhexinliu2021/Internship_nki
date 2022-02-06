# Dataclass to perform PCA on gene expression values
# predicted from enformer.
# Data: 5/2/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCA_gene:
    def __init__(self, **keypar):  # keypar: { 'name' :gene_name,
                                   # 'df': dataframe (Depmap sample ids as indexx,
                                   #  gene names as col names}
        self.name = keypar['name']
        self.df = keypar['df']

    def normalize(self):
        # normalize to mean~0, std~1
        self.x =  StandardScaler().fit_transform(self.df.values)
        return self.x

    def do_pca(self, ex_var = 0.8):
        self.pca = PCA(ex_var)
        self.pca.fit(self.x)
        self.cps, self.ex_var = self.pca.n_components_, self.pca.explained_variance_ratio_
        return (self.cps, self.ex_var)

    def transform(self):
        try:
            self.transformed = self.pca.fit_transform(self.x)
        except (Exception )as e:
            print(e)
            print(f'you have to fit the pca model first')
        else:
            return(self.transformed ) # np.array


def Ex_var(df, ex_var = 0.8):
    x = StandardScaler().fit_transform(df.values)
    pca = PCA(ex_var)
    pca.fit(x)
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    return(pca.n_components_, pca.explained_variance_ratio_)