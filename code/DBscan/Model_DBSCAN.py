# Build ElasticNet on the clustering of DBSCAN of
# T-SNE components.

import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('/home/lzhexin/job_scripts/Enformer/code/PCA')
from ElasticNetModules import EN_cv_in, out_loop, drug_model


file_path = '/Gen_exp_1.0'
drug_file = "/CCLE/Drug_sensitivity_AUC.csv"
sra_info = "/CCLE/sra_result.csv"
run_info = "/CCLE/SraRunInfo.csv"
sample_info_file = "/CCLE/sample_info.csv"
PCA_file = '/PCAs'

def main():
    df = pd.read_csv(f'{PCA_file}/DB_clusters.csv', index_col=0)
    target_df = pd.read_csv(drug_file, index_col=0)

    # remove drug columns with multiple treatments.
    target_df =  target_df.loc[: , target_df.columns.map(lambda x:x.split('(CTRP')[0].strip().count(':') != 2)]
    sample_df = pd.read_csv(sample_info_file)
    run_df = pd.read_csv(run_info, index_col='SampleName')

    mapping = sample_df.loc[:, ['DepMap_ID', 'CCLE_Name']].set_index(['CCLE_Name'])
    mapping = mapping.loc[run_df.index, :]
    mapping['Run'] = run_df.Run

    consensus_runs = mapping.DepMap_ID.isin(target_df.index).values
    mapping = mapping.loc[consensus_runs, :]

    target_df = target_df.loc[mapping.DepMap_ID, :]

    target_df = target_df.set_index([target_df.index, mapping.reset_index().
                                    set_index('DepMap_ID').loc[target_df.index, :].Run])

    DF = df.loc[mapping.Run]
    DF = DF.set_index([mapping.reset_index().set_index(['Run']).
                      loc[DF.index, :].DepMap_ID, DF.index])

    #build models
    # get rid of drugs with too many missing values.
    thresh  = int(target_df.shape[0]*(0.80)) ## at least 80% of Non-na value in each drug
    drug_df_ = target_df.dropna(axis = 1, thresh = thresh)
    print(f'before filtering drugs: the number of drugs is {target_df.shape[1]}',  flush=True)
    print(f'after filtering, the number of drugs is {drug_df_.shape[1]}',  flush=True)
    print(f'Shape of drug data: {drug_df_.shape}',  flush=True)

    comp_index = dg_index - 1
    target = drug_df_.iloc[:, comp_index]
    pre = drug_model(DF, target, out_path=out_dir, norm_bool=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_index", type=int,
                        help="target index (1-based). 1-359")
    parser.add_argument("-o", "--output", type=str,
                        help="output file's dir ")
    parser.add_argument("-p", "--r_path", type=str,
                        help="obsolute path on scratch")

    args = parser.parse_args()

    dg_index = args.target_index
    out_dir = args.output
    r_path = args.r_path

    file_path = f'{r_path}/Gen_exp_1.0'
    drug_file = f"{r_path}/CCLE/Drug_sensitivity_AUC.csv"
    sra_info = f"{r_path}/CCLE/sra_result.csv"
    run_info = f"{r_path}/CCLE/SraRunInfo.csv"
    sample_info_file = f"{r_path}/CCLE/sample_info.csv"
    PCA_file = f'{r_path}/{PCA_file}'
    ##############

    print('Python program started', flush=True)
    main()
