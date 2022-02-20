# This is the implementation of
# modeling gene expression using
# ElasticNet to predict drug response
# (AUC) in CCLE dataset.
# Data: 20/2/2022

# usage: ElasticNet_geneEXp.py [-h] [-t TARGET_INDEX] [-o OUTPUT] [-p R_PATH]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -t TARGET_INDEX, --target_index TARGET_INDEX
#                         target index (1-based). 1-358
#   -o OUTPUT, --output OUTPUT
#                         output file's dir
#   -p R_PATH, --r_path R_PATH
#                         obsolute path on scratch



import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import math
warnings.filterwarnings('always')
# add PCA models' path to the sys.path
model_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.append(model_dir)
from PCA.ElasticNetModules import drug_model




def main():
    #Read files
    CCLE_expression_log2 = pd.read_csv(expression_path)
    Con_df = pd.read_csv(Con_file)
    target_df = pd.read_csv(target_path)
    print('Reading files done', flush=True)

    # Create matrix of genomics features.
    # select gene list
    Cos_gene_list = Con_df[Con_df.columns[0]].values
    CCLE_genes = CCLE_expression_log2.columns[1:].map(lambda x: x.split(' ')[0])

    # CCLE_genes.isin(Cos_gene_list)
    df = CCLE_expression_log2.set_index(CCLE_expression_log2.columns[0]).loc[:, CCLE_genes.isin(Cos_gene_list)]

    ### Create target
    df = df.loc[df.index.isin(target_df.iloc[:,0]),:]
    target_df = target_df.loc[target_df.iloc[:,0].isin( df.index),:].set_index(target_df.columns[0])

    target_df = target_df.loc[: , target_df.columns.map(lambda x:x.split('(CTRP')[0].strip().count(':') != 2)]

    " take drugs with less than 20% of missing values"
    target_df = target_df.dropna(axis = 1, thresh= target_df.shape[0] * 0.8)

    print(f'dimention of genomics features matrix: {df.shape}', flush=True)
    print(f'dimention of genomics features matrix: {target_df.shape}', flush=True)



    # build model for each drug.
    """Training schema: 1. Bootstrapping 80% of the data (FUNCTION "drug_model")
                    2. Outter loop to split resampled data into 5-folds for training and 
                         validation (implemented in FUNCTION "out_loop" )
                    3. During each iteration of training, apply inner 10-fold loop cross 
                       validation to select best l1 ratio and alpha pair. (FUNCTION EN_cv_in)
                    4. Vaildate on the 5th fold testing set. 
    """

    comp_index = dg_index - 1
    target = target_df.iloc[:, comp_index]
    pre = drug_model(df, target, out_path=out_dir,  norm_bool = False, l1_range = np.linspace(start=0.2, stop=1.0, num=5),
                     alpha_list =  np.array([math.exp(i) for i in np.arange(-15, 5, 2)]) )


if __name__ == '__main__':
    # file paths
    expression_path = '/CCLE/CCLE_expression_full.csv'
    Con_file = '/CCLE/mutation_files/Census_allSat.csv'
    target_path = "/CCLE/Drug_sensitivity_AUC.csv"



    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_index", type=int,
                        help="target index (1-based). 1-358")
    parser.add_argument("-o", "--output", type=str,
                        help="output file's dir ")
    parser.add_argument("-p", "--r_path", type=str,
                        help="obsolute path on scratch")

    args = parser.parse_args()

    dg_index = args.target_index
    out_dir = args.output
    r_path = args.r_path

    ########################
    expression_path = f'{r_path}{expression_path}'
    Con_file = f'{r_path}{Con_file}'
    target_path = f'{r_path}{target_path}'

    print('Python program started', flush=True)
    main()



