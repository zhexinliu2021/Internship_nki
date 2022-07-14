import pandas as pd
import argparse
import numpy as np
import sys
sys.path.append('/home/lzhexin/job_scripts/Enformer/code/PCA')
from ElasticNetModules import EN_cv_in, out_loop, drug_model, select_fea, select_para
from sklearn.linear_model import ElasticNet
import math

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


    # significant drugs
    sig_drugs = pd.Series(['TW-37 (CTRP:609596)', 'BI-2536 (CTRP:347813)', 'vemurafenib (CTRP:649420)',
     'CI-976 (CTRP:375390)', 'valdecoxib (CTRP:32372)', 'gossypol (CTRP:25036)'], index = [238, 46, 286, 69, 39, 16])

    sig_drug_df_ = drug_df_.loc[:, sig_drugs.values]
    comp_index = dg_index - 1
    target = sig_drug_df_.iloc[:, comp_index]

    print("processing data done", flush=True)

    #STEP1: select best parameters
    output_item = select_para(DF, target,  norm_bool=True,
                              n_cv=50,
                              l1_range=np.linspace(start=0.2, stop=1.0, num=5),
                              alpha_list=np.array([math.exp(i) for i in np.arange(-15, 5, 2)])
                              )
    op_l1_ratio, op_alpha = output_item['arguments']
    X, Y = output_item['processed_x'], output_item['processed_y']
    print('selection of best parameters done.')
    print(f'best l1_ratio: {op_l1_ratio} \n best alpha: {op_alpha}')


    w_list = []
    n_cv = 200
    for i in range(n_cv):
        w = select_fea(X=X, Y=Y, alpha=op_alpha, l1_ratio=op_l1_ratio, random_state=i)
        w_list.append(w)
        print(f'CV {i} done, total {n_cv}')
    w_list = pd.DataFrame(w_list, columns=DF.columns)

    w_list.to_csv(f'{out_dir}/{target.name}.csv')

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
