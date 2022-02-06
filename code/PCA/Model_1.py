# 1. Do PCA for each gene across 329 samples.
# 2. Concat gene profiles
# 3. Build linear model (elasticNet) to predict the drug AUCs
# Data: 5/2/2022


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import copy
import PCAmodules
from ElasticNetModules import EN_cv_in, out_loop, drug_model
import argparse
pd.options.mode.chained_assignment = None  # default='warn

file_path = '/Gen_exp'
drug_file = "/CCLE/Drug_sensitivity_AUC.csv"
sra_info = "/CCLE/sra_result.csv"
run_info = "/CCLE/SraRunInfo.csv"
sample_info_file = "/CCLE/sample_info.csv"
PCA_file = '/PCAs'

def gene_generator(file_path):
    for i in os.listdir(file_path):
        if i.endswith('.csv'):
            yield {'name' : i.split('.')[0],
                   'df' : pd.read_csv(f'{file_path}/{i}', index_col=0)}



def main(PCA_ = False):

    if PCA_:
        # STEP 1: Conduct PCA operation for all genes
        df = pd.DataFrame()
        ex_var = 0.8
        for index, gene_item in enumerate(gene_generator(file_path)):
            # load data
            GENE = PCAmodules.PCA_gene(**gene_item)
            print(f'processing {GENE.name}, dimension: {GENE.df.shape}.',flush=True)
            # normalize the data
            scaled_x = GENE.normalize()

            # conduct pca
            ncps, exlained_var_ratio = GENE.do_pca(ex_var=ex_var)
            # transform the data to number of components that satisfy the
            # variance.
            transformed_x = GENE.transform()  # np.array
            print(f"dimension after PCA: {transformed_x.shape}", flush=True)

            df_ = pd.DataFrame(transformed_x, columns=[f'{GENE.name}_cp_{i + 1}' for i in range(ncps)],
                               index=gene_item['df'].index)
            df = pd.concat([df, df_], axis=1)

        #df.to_csv(f'/home/lzhexin/job_scripts/Enformer/files/PCA_{ex_var}.csv')
    elif not PCA_:
        df = pd.read_csv(f'{PCA_file}/PCA_0.8.csv', index_col = 0)

    # STEP 2:
    # build ElasticNet module to predict target
    # 2.1 upload x and y
    # eliminate drug repsonse with multiple drug treatment.
    drug_df = pd.read_csv(drug_file, index_col=0)
    drug_df = drug_df.iloc[:,
              drug_df.columns.map(lambda x: len(x.split('(CTRP')[0].strip().split(' ')[0].split(':')) == 1)]
    drug_df.columns = drug_df.columns.map(lambda x: x.split('(CTRP')[0].strip())
    print(f'dimension of processed drug labels: {drug_df.shape}', flush=True)

    # Read sample_infor to map Depmap ID to CCLE name
    sample_info = pd.read_csv(sample_info_file)
    # np.isin(sample_info.loc[:,'DepMap_ID'], drug_df.index)
    mapping = sample_info.loc[np.isin(sample_info.loc[:, 'DepMap_ID'], drug_df.index), ['DepMap_ID', 'CCLE_Name']]

    # Read run_infor to map Run name to CCLE name
    run_info_ = pd.read_csv(run_info)
    run_info_.set_index(['Run'], inplace=True)
    run_info_ = run_info_.loc[:, 'SampleName']

    # Take the intersection of drug file and 329-sample file.
    consens_id = mapping.loc[np.isin(mapping.CCLE_Name, run_info_.values), :]
    consens_id.loc[:, 'Run'] = run_info_[np.isin(run_info_.values, mapping.CCLE_Name)].index.copy()

    drug_df = drug_df.loc[consens_id.DepMap_ID, :]
    drug_df.reset_index(inplace=True)
    drug_df.loc[:, 'Run'] = consens_id.set_index('DepMap_ID').loc[drug_df.loc[:, 'index'], :].Run.values
    drug_df.set_index(['index', 'Run'], inplace=True)

    df = df.loc[consens_id.Run, :].reset_index()
    df.loc[:, 'DepMap_ID'] = consens_id.DepMap_ID.values
    df.set_index(['DepMap_ID', 'index'], inplace=True)
    df_ = df

    # 2.2
    # Build models
    # get rid of drugs with too many missing values.
    thresh = int(df.shape[0] * (0.80))  ## at least 80% of Non-na value in each drug
    drug_df_ = drug_df.dropna(axis=1, thresh=thresh)
    print(f'before filtering drugs: the number of drugs is {drug_df.shape[1]}', flush=True)
    print(f'after filtering, the number of drugs is {drug_df_.shape[1]}', flush=True)
    print(f'Shape of drug data: {drug_df_.shape}', flush=True)

    ### build model for each drug.
    """Training schema: 1. Bootstrapping 80% of the data (FUNCTION "drug_model")
                        2. Outter loop to split resampled data into 5-folds for training and 
                             validation (implemented in FUNCTION "out_loop" )
                        3. During each iteration of training, apply inner 10-fold loop cross 
                           validation to select best l1 ratio and alpha pair. (FUNCTION EN_cv_in)
                        4. Vaildate on the 5th fold testing set. 
    """
    comp_index = dg_index -1
    target = drug_df_.iloc[:, comp_index]
    pre = drug_model(df_, target, out_path=out_dir)
    ###


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

    file_path = f'{r_path}/Gen_exp'
    drug_file = f"{r_path}/CCLE/Drug_sensitivity_AUC.csv"
    sra_info = f"{r_path}/CCLE/sra_result.csv"
    run_info = f"{r_path}/CCLE/SraRunInfo.csv"
    sample_info_file = f"{r_path}/CCLE/sample_info.csv"
    PCA_file = f'{r_path}/{PCA_file}'
    ##############

    print('Python program started', flush=True)
    main(PCA_ = False)
