# This is the implementation of modeling coding mutations
# to predict drug response (AUC) in CCLE dataset.
# Data: 1/2/2022




import pandas as pd
import numpy as np 
import os
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings
from tools import write_file
import argparse
import tools
import warnings
warnings.filterwarnings('always')



# In[2]:


# SET FILE PATHS

path = '/home/lzhexin/job_scripts/baseline/CCLE/mutation_files/' # dir contains mutation data

file_list = ['damaging','hotspot','nonconserving','otherconserving'] # four mutation vectors
drug_path = "/CCLE/Drug_sensitivity_AUC.csv"
#out_path = '/Users/jerryliu/jerry_jupyter/internship/files/output/' #dir to store predictions
Con_file='/CCLE/mutation_files/Census_allSat.csv' # Cosmic gene
                                                                                             # list
sample_info = "/CCLE/sample_info.csv"


# In[3]:


# **Source of the data**:
# Mutation vectors: https://depmap.org/portal/download/    (version: Depmap Public 2021Q4 )
# Drug response data: https://depmap.org/portal/download/   (CUSTOMS DOWNLOADS -> compound 
#                                                               -> Drug sensitivity AUC (CTD^2)
# Cancer gene census: https://cancer.sanger.ac.uk/cosmic/census?genome=37    (in total 729 genes)


# In[3]:


#read drug sensitivity files. 
def read_drug(drug_path):
    """ READ DRUG RESPONSE FILE
    return a matrix with cell line ids as row index
    """
    drug_data = pd.read_csv(drug_path)
    drug_data = drug_data.set_index(drug_data.columns[0])
    return(drug_data)


# In[4]:


#funtion to merge two mutation files.
def merge_mt(m_f1, m_f2, drug_id, gene_filter = False):
    """ INPUT: [m_f1, m_f2]:  two mutation data matrices 
                    drug_id:  cell line ids in drug response data 
                    gene_filter: * If False:  Merged cell lines =  (m_f1 | drug cl) & (m_f2 | drug cl) 
                                 * If True: merged cell lines = (m_f1 | m_f2) 
                                 * by default (False), the idea is to take union of cell lines from two 
                                   mutation datasets with valid drug reponse targets. For column, we take 
                                   union of genes from two sets. 
        OUTPUT: merged data frame
        FUNCTION: merge two mutation vectors, if a gene has a mutation record (1) in either of the 
                  two vectors, then it is a 1 in the merged data set.
    """
    
    # input: dataframe with index as cell line id. 
    # output: merged dataframe
    #cell lines: m_f1 & m_f2;  cancer genes: m_f1 | m_f2.
    if not gene_filter:
        cl_id = set.union(set.intersection(set(m_f1.index), set(drug_id)), 
                         set.intersection(set(m_f2.index), set(drug_id)))
    else:
        cl_id = set.intersection(set(m_f1.index), set(m_f2.index))
        
    gene_id = set.union(set(m_f1.columns), set(m_f2.columns))

    m_f1 = m_f1.reindex(cl_id, columns = gene_id, fill_value = 0 )
    m_f2 = m_f2.reindex(cl_id, columns = gene_id, fill_value = 0)
    #combine the two files.
    m_comb = pd.DataFrame(np.where(m_f1 == 0, m_f2, m_f1), index = cl_id, columns=gene_id)
    
    return(m_comb)


# In[5]:



def EN_cv_in(x_train, y_train, fold, alpha, l1_ratio):
        """
        INPUT: training set. (each alpha, l1_ratio parameter)
        OUTPUT: averge spearman correlation score of 10 results
        FUNCTION: conduct 10 cross validation on the 80 % of resampled dataset
                  to determine the best l1_ratio and alpha pair.
        """
        #alpha = p1, l1_ratio = p2
        kf_in = KFold(n_splits = fold)
        cv_perf = np.array([]) 
        for random_, (train_in, test_in) in enumerate(kf_in.split(x_train)):

            x_train_in, x_test_in, y_train_in, y_test_in = x_train[train_in], x_train[test_in], y_train[train_in], y_train[test_in]

            #build model with each combination of parameters. 
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                regr = ElasticNet(random_state = random_, alpha= alpha, l1_ratio=l1_ratio, 
                                  fit_intercept = False, max_iter= 3000)
                regr.fit(x_train_in, y_train_in)
                
            y_pre = regr.predict(x_test_in)
            #perforamance matrix of the model is the Pearson's corrlation coeeficient r. 
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = pd.Series(y_pre).corr(pd.Series(y_test_in), method = 'spearman')
                cv_perf = np.append(cv_perf,[r])
                del regr
        return(np.mean(cv_perf))

    
def out_loop(x, y_total, cl_id, out_cv_fold= 5):
    ''' 5 fold cv (outter-loop)
    # x: (#_(80%)C, #_genes); y: (#_(80%)C,)
    FUNCTION: conduct 5-fold cross validation on the resampled data. In each iteration, conduct
              10 fold CV on the training set (EN_cv_in function) to do parameter search and select the l1
              ratio and alpha with highest validation score on the 10th fold testing set
    '''
    kf = KFold(n_splits = out_cv_fold)
    pre_y_array = dict(list(zip(cl_id,[ [] for _ in range(len(cl_id))])))
    per_test_list = []
    for train, test in kf.split(x):
        x_train, x_test, y_train, y_test = x.values[train], x.values[test], y_total.values[train], y_total.values[test]
        train_label, test_label = x.index[train], x.index[test]
        # 10-fold in x_train (inner-loop). 
        # each pair of alpha and l1_ratio, do cross training to sellect best pair. 

        l1_ratio_list = np.linspace(start = 0.2, stop = 1.0, num = 5) #10 values
        alpha_list =  np.array([math.exp(i) for i in np.arange(-6, 5, 0.8)] ) # 250 values
        #alpha_list =  np.array([math.exp(i) for i in np.arange(-8,5,0.8)]) #  values
        para_matrix = {(l1_ratio, alpha):0 for l1_ratio in l1_ratio_list for alpha in alpha_list}

        for (l1_ratio, alpha) in para_matrix:
            #do ten fold cv to sellect best parameter pairs. 
            ave_per = EN_cv_in(x_train= x_train, y_train = y_train, fold = 8, alpha = alpha, l1_ratio= l1_ratio)
            para_matrix[(l1_ratio, alpha)] = ave_per
        #print(para_matrix)
        
        #return the best alpha-l1-ratio pair.
        op_l1_ratio, op_alpha = pd.Series(para_matrix).idxmax() 
        print('the best validation performance is: ', para_matrix[(op_l1_ratio, op_alpha)],  flush=True)
        
        #predict on the 5th-fold set (testing set)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regr = ElasticNet(random_state=0, alpha= op_alpha, l1_ratio = l1_ratio, 
                              fit_intercept = False, max_iter= 3000)
            regr.fit(x_train, y_train)
            y_pre = regr.predict(x_test)
            
        if len(np.unique(y_pre) ) == 1: print('constant prediction.', flush=True)
        del regr
        #add results to the storage dictionary. 
        for index, value in zip(test_label, y_pre): pre_y_array[index].append(value)
        per_test = pd.Series(y_pre).corr(pd.Series(y_test),method = 'spearman') 
        per_test_list.append(per_test)
        print('performance on the test set is:', per_test, flush=True)
        # plt.scatter(y_test, y_pre)
        # plt.xlabel('test set')
        # plt.ylabel('prediction')
        # plt.show()
        #break
    #add the prediction to the list.
    pre_y_array = pd.Series(pre_y_array).reindex(cl_id)
    
    return(pre_y_array, per_test_list)


def drug_model(m_file, target, out_path):
    """
    INPUT: m_file mutation matrix with unpreprocessed binery values.
    OUTPUT: mean prediction values of 10 iterations (TYPE: pd.Sereis).
    FUNCTION: conduct normalization + 10 times of 5-fold cross validation
    """
    #eliminate cell lines with nan drug sensitivity data.
    drug_name = target.name
    Y = target.dropna()
    m_file_notna = m_file.loc[Y.index,:]

    #normalize the each col to have ~0 mean and ~1 sd. 
    # old normalization method:
    # norm_m_file = (m_file_notna - m_file_notna.mean(axis = 0)) / m_file_notna.std(axis = 0)
    norm_m_file = tools.scal_matrix(m_file_notna)
    
    print('### normalize with mean=0, std = 1.', flush=True)
    norm_m_file.dropna(axis = 1, inplace = True)
    print('number of features selected {}'.format(norm_m_file.shape[1]), flush=True)
    X, Y_total = norm_m_file, Y
    
    ### EN model
    ## bootstraping (80% of data) for 10 times. 
    ## each time with 80% data for training and remaining 20% for
    ## testing (5 fold CV).
    cl_id = norm_m_file.index
    Y_pre_array = pd.Series(dict(list(zip(cl_id, [[] for i in range(cl_id.shape[0])]))),
                            index = cl_id, name = drug_name)
    cv_result_list = []
    for i in range(10):
        print('iteration {} begains'.format(i), flush=True)
        #resample 80% of data
        x, y_total = resample(X, Y_total, replace = False, n_samples = int(X.shape[0]*0.9), random_state = i)
        #print(X.isna().sum().sum())
        #print(y_total.isna().sum().sum())

        #outter loop with 5 fold cv.
        prediction, cv_list = out_loop(x = x, y_total= y_total, cl_id = cl_id, out_cv_fold=5 )
        cv_result_list.extend(cv_list)
        print('iter {}, outter loop finished. '.format(i), flush=True)
        
        for i in prediction.index: Y_pre_array[i].extend(prediction[i])

    
    ##calculate the mean of 10 iterations. 
    #Y_pre_array = Y_pre_array.map(np.mean)
    tools.write_file(Y_pre_array, 'comp_{}.csv'.format(drug_name), out_path)
    # list of cross validation results. 
    return(cv_result_list)


# In[6]:
def map_id(CCLE_name):
    DepMap_ID = sample_info.loc[:,"DepMap_ID"][sample_info.CCLE_Name.values == CCLE_name]
    if len(DepMap_ID) == 0:
        return np.NAN
    return DepMap_ID.values[0]


def main():
    #STEP 1. ---> load data <---
    #sample_info = pd.read_csv(sample_info)
    m_f1, m_f2, m_f3, m_f4  = file_list
    m_f1 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f1 + '.csv', index_col = 0)
    m_f2 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f2 + '.csv', index_col = 0)
    m_f3 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f3 + '.csv', index_col = 0)
    m_f4 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f4 + '.csv', index_col = 0)
    print('#### reading files completed', flush=True)


    # In[7]:


    #STEP 2.1 ---> define target cancer genes <---
    gene_bankmap = {gene.split(' ')[0]:gene.split(' ')[1]
    for gene in tools.get_cons(m_f1.columns, m_f2.columns, m_f3.columns, method='union')}

    Con_genes = pd.read_csv(Con_file).loc[:,'Gene Symbol'] # Get Cosmis gene list
    gene_lists =  [ m_file.columns.map(lambda x:x.split(' ')[0])  # Get union of three gene lists
    for m_file in [m_f1, m_f2, m_f3] ]                        # in mutation files. Here we use first
                                                              # three (m_f1, m_f2, m_f3) mutation vectors.
    final_genes = tools.get_cons(tools.get_cons(*gene_lists, method='union'),
               Con_genes, method = 'inter')  # We take intersection of the Cosmic and CCLE gene list.
    final_genes = [f"{gene} {gene_bankmap[gene]}" for gene in final_genes ]
    #print(f'number of final genes is {len(final_genes)}')


    # In[8]:



    # In[22]:


    # STEP 2.2 process drug data

    # drug_data = read_drug(drug_path = drug_path)
    # drug_data = drug_data.reset_index().loc[:,["CCLE Cell Line Name", "Compound", "ActArea" ]].set_index(["CCLE Cell Line Name", "Compound"])
    # drug_data = drug_data.sort_index().unstack()

    # # step 2.2.1 map the CCLE names to DepMap_ID
    # #sample_info.loc[:,['CCLE_Name', 'DepMap_ID'] ]
    # drug_data.columns = drug_data.columns.map(lambda x:x[1])
    # drug_data


    # In[23]:


    # drug_data.set_index(drug_data.index.map(map_id), inplace=True)

    # drug_data = drug_data.loc[drug_data.index.notna(),:]


    # In[24]:


    drug_data =  read_drug(drug_path = drug_path)
    # Only select single-drug treatment.
    drug_data = drug_data.loc[:,drug_data.columns[drug_data.columns.map(lambda x: len(x.split(' ')[0].split(':'))) == 1]]
    drug_data.columns = drug_data.columns.map(lambda x: x.split(' ')[0])
    print(f'drug data processing done', flush=True)
    print(f'shape of drug data is {drug_data.shape}', flush=True)

    # In[14]:


    # STEP 2.2 ---> concact three mutation matrix instead of mergeing genes ---
    inter_cls = set.intersection(set(m_f1.index), set(m_f2.index), set(m_f3.index), set(drug_data.index))
    m_f1_ = tools.trans_genes(m_f1, Con_genes, gene_bankmap).loc[inter_cls,:]
    m_f2_ = tools.trans_genes(m_f2, Con_genes, gene_bankmap).loc[inter_cls,:]
    m_f3_ = tools.trans_genes(m_f3, Con_genes, gene_bankmap).loc[inter_cls,:]
    m_file = pd.concat([m_f1_, m_f2_, m_f3_], axis=1)
    (m_file == 1).sum().plot.hist(bins = 50)
    print(m_file.shape, flush=True)


    # In[ ]:


    # STEP 2.2 ---> merge mutation files <---
    # m_file = merge_mt(merge_mt(m_f1, m_f2, drug_id=drug_data.index), m_f3,
    #                   drug_id= drug_data.index,gene_filter=False)      # Merge m_f1, m_f2, m_f3 CCLE
    # #                                                                    # mutation metrices.
    #
    # print('#### merging mutation files completed')
    #
    # m_file = m_file.loc[:,final_genes] # Now we take selected 710 genes.
    # print(m_file.shape)
    # m_file.head()


    # In[22]:


    # STEP 3. ---> process drug labels <---
    # get rid of drugs with too many missing values.
    thresh = int(m_file.shape[0] * (0.60))  ## at least 80% of Non-na value in each drug
    drug_df = drug_data.loc[m_file.index, :].dropna(axis=1, thresh=thresh)
    print(f'before filtering drugs: the number of drugs is {drug_data.shape[1]}', flush=True)
    print(f'after filtering, the number of drugs is {drug_df.shape[1]}', flush=True)
    print(f'Shape of drug data: {drug_df.shape}', flush=True)
    #drug_df.head()


    # In[125]:




    #build model for each drug.
    """Training schema: 1. Bootstrapping 80% of the data (FUNCTION "drug_model")
                    2. Outter loop to split resampled data into 5-folds for training and 
                         validation (implemented in FUNCTION "out_loop" )
                    3. During each iteration of training, apply inner 10-fold loop cross 
                       validation to select best l1 ratio and alpha pair. (FUNCTION EN_cv_in)
                    4. Vaildate on the 5th fold testing set. 
    """

    comp_index = dg_index -1
    target = drug_df.iloc[:,comp_index]
    pre = drug_model(m_file, target, out_path = out_dir)



# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target_index", type=int,
                        help= "target index (1-based). 1-494")
    parser.add_argument("-o","--output", type = str,
                        help= "output file's dir ")
    parser.add_argument("-p", "--r_path", type = str,
                        help= "obsolute path on scratch")

    args = parser.parse_args()
    
    dg_index = args.target_index
    out_dir = args.output
    r_path = args.r_path
    ##### Define paths
    path = f'{r_path}/CCLE/mutation_files/'
    drug_path = f'{r_path}{drug_path}'
    Con_file = f'{r_path}{Con_file}'
    ######
    print('Python program started', flush=True)
    main()

