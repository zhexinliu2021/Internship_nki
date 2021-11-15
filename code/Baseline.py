#!/usr/bin/env python
# coding: utf-8

# In[146]:


## prepare data


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
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
import importlib
from tools import write_file
import multiprocessing
import tools
import importlib
importlib.reload(tools)


# In[2]:


#load four mutation files.
path = '/Users/jerryliu/Documents/Vu_uva/internship/CCLE/mutation_files/'
file_list = ['damaging','hotspot','nonconserving','otherconserving']
drug_path = '/Users/jerryliu/Documents/Vu_uva/internship/CCLE/primary-screen-replicate-collapsed-logfold-change.csv'
out_path = '/Users/jerryliu/jerry_jupyter/internship/output/'


# In[3]:


#read drug sensitivity files. 
def read_drug(drug_path):
    drug_data = pd.read_csv(drug_path)
    drug_data = drug_data.set_index(drug_data.columns[0])
    return(drug_data)


# In[4]:


#funtion to merge two mutation files.
def merge_mt(m_f1, m_f2, drug_id, gene_filter = False):
    # input: dataframe with index as cell line id. 
    # output: merged dataframe
    #cell lines: m_f1 & m_f2;  cancer genes: m_f1 | m_f2.
    if not gene_filter:
        cl_id = set.intersection(set(m_f1.index), set(m_f2.index), set(drug_id))
    else:
        cl_id = set.intersection(set(m_f1.index), set(m_f2.index))
        
    gene_id = set.union(set(m_f1.columns), set(m_f2.columns))

    m_f1 = m_f1.reindex(cl_id, columns = gene_id, fill_value = 0 )
    m_f2 = m_f2.reindex(cl_id, columns = gene_id, fill_value = 0)
    #combine the two files.
    m_comb = pd.DataFrame(np.where(m_f1 == 0, m_f2, m_f1), index = cl_id, columns=gene_id)
    
    return(m_comb)


# In[5]:


def filter_genes(file, thresh = 0.1):
    """
    thresh: at least ## of cell lines are mutated and are kept. 
    input: list of dataframes of mutated genes.
    output: list of most mutated genes. 
    """
    gene_bool = file.apply(lambda x: True if (x == 1).sum()/len(x) >= thresh else False).values
    return(file.columns[gene_bool])
    


# In[15]:



def EN_cv_in(x_train, y_train, fold, alpha, l1_ratio):
        """
        input: training set. (each alpha, l1_ratio parameter)
        output: averge spearman correlation score of 10 results
        """
        #alpha = p1, l1_ratio = p2
        kf_in = KFold(n_splits = fold)
        cv_perf = np.array([]) 
        for train_in, test_in in kf_in.split(x_train):

            x_train_in, x_test_in, y_train_in, y_test_in = x_train[train_in], x_train[test_in], y_train[train_in], y_train[test_in]

            #build model with each combination of parameters. 
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                regr = ElasticNet(random_state = 0, alpha= alpha, l1_ratio=l1_ratio, fit_intercept = False)
                regr.fit(x_train_in, y_train_in)
                
            y_pre = regr.predict(x_test_in)
            #perforamance matrix of the model is the Pearson's corrlation coeeficient r. 
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = pd.Series(y_pre).corr(pd.Series(y_test_in), method = 'spearman')
                cv_perf = np.append(cv_perf,[r])
                del regr
        return(np.mean(cv_perf))

    
def out_loop (x, y_total, cl_id, out_cv_fold= 5):
    ''' 5 fold cv (outter-loop)
    # x: (#_(80%)C, #_genes); y: (#_(80%)C,)
    '''
    kf = KFold(n_splits = out_cv_fold)
    pre_y_array = dict(list(zip(cl_id,[ [] for _ in range(len(cl_id))])))
    per_test_list = []
    for train, test in kf.split(x):
        x_train, x_test, y_train, y_test = x.values[train], x.values[test], y_total.values[train], y_total.values[test]
        train_label, test_label = x.index[train], x.index[test]
        # 10-fold in x_train (inner-loop). 
        # each pair of alpha and l1_ratio, do cross training to sellect best pair. 

        l1_ratio_list = np.linspace(start = 0.2, stop = 1.0, num = 10) #10 values
        alpha_list =  np.array([math.exp(i) for i in np.arange(-16,5,2)]) # 250 values
        #alpha_list =  np.array([math.exp(i) for i in np.arange(-8,5,0.8)]) #  values
        para_matrix = {(l1_ratio, alpha):0 for l1_ratio in l1_ratio_list for alpha in alpha_list}

        for (l1_ratio, alpha) in para_matrix:
            #do ten fold cv to sellect best parameter pairs. 
            ave_per = EN_cv_in(x_train= x_train, y_train = y_train, fold = 10, alpha = alpha, l1_ratio= l1_ratio)
            para_matrix[(l1_ratio, alpha)] = ave_per
        print(para_matrix)
        
        #return the best alpha-l1-ratio pair.
        op_l1_ratio, op_alpha = pd.Series(para_matrix).idxmax() 
        print('the best validation performance is: ', para_matrix[(op_l1_ratio, op_alpha)])
        
        #predict on the 5th-fold set (testing set)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regr = ElasticNet(random_state=0, alpha= op_alpha, l1_ratio = l1_ratio, fit_intercept = False)
            regr.fit(x_train, y_train)
            y_pre = regr.predict(x_test)
            
        if len(np.unique(y_pre) ) == 1: print('constant prediction.')
        del regr
        #add results to the storage dictionary. 
        for index, value in zip(test_label, y_pre): pre_y_array[index].append(value)
        per_test = pd.Series(y_pre).corr(pd.Series(y_test),method = 'spearman') 
        per_test_list.append(per_test)
        print('performance on the test set is:', per_test)
        plt.scatter(y_test, y_pre)
        plt.xlabel('test set')
        plt.ylabel('prediction')
        plt.show()
        #break
    #add the prediction to the list.
    pre_y_array = pd.Series(pre_y_array).reindex(cl_id)
    
    return(pre_y_array, per_test_list)

def drug_model(comp_index, m_file,  drug_data, out_path):
    """
    input: m_file mutation matrix with unpreprocessed binery values.
    output: mean prediction values of 10 iterations in pd.Sereis type.
    """
    #eliminate cell lines with nan dryg sensitivity data.
    #comp_index = 0
#     print(comp_index)
#     print(m_file.shape)
    
    valid_cl_index = drug_data.loc[m_file.index,:].iloc[:,comp_index].dropna().index
    Y = drug_data.iloc[:, comp_index][valid_cl_index]
    drug_name = drug_data.columns[comp_index]
    print('drug_name is: ',drug_name)
    print('number of missing values in Y: ', Y.isnull().sum())
    m_file_notna = m_file.loc[valid_cl_index,:]
    
    #normalize the each col to have ~0 mean and ~1 sd. 
    norm_m_file = (m_file_notna - m_file_notna.mean(axis = 0)) / m_file_notna.std(axis = 0)
    print(norm_m_file.sum())
    
    print('### normalize with mean=0, std = 1.')
    norm_m_file.dropna(axis = 1, inplace = True)
    
#     #Use features that have pearson correlation with Y over 0.1.
#     cor_number = norm_m_file.corrwith(Y, axis = 0, method = 'pearson')
#     #number of genes that have correlation over 0.1
#     #sum(np.abs(cor_number) >= 0.1)
#     norm_m_file = norm_m_file.loc[:, np.abs(cor_number) >= 0.1]
#     #print('number of features selected {}'.format(sum(np.abs(cor_number) >= 0.1)))
    
    
    print('number of features selected {}'.format(norm_m_file.shape[1]))
    X, Y_total = norm_m_file, Y
    
    ### EN model
    ## bootstraping (80% of data) for 10 times. 
    ## each time with 80% data for training, 10% for cross validataion, and remaining 10% for
    ## testing.
    cl_id = norm_m_file.index
    Y_pre_array = pd.Series(dict(list(zip(cl_id, [[] for i in range(cl_id.shape[0])]))),
                            index = cl_id, name = drug_name)
    cv_result_list = []
    for i in tqdm(range(10)):
        print('iteration {} begains'.format(i))
        x, y_total = resample(X, Y_total, replace = False, n_samples = int(X.shape[0]*0.8), random_state = i)
        print(X.isna().sum().sum())
        print(y_total.isna().sum().sum())

        #outter loop with 5 fold cv.
        prediction, cv_list = out_loop(x = x, y_total= y_total, cl_id = cl_id, out_cv_fold=5 )
        cv_result_list.extend(cv_list)
        print('iter {}, outter loop finished. '.format(i))
        
        for i in prediction.index: Y_pre_array[i].extend(prediction[i])

    
    ##calculate the mean of 10 iterations. 
    Y_pre_array = Y_pre_array.map(np.mean)
    tools.write_file(Y_pre_array, 'comp_{}.csv'.format(comp_index), out_path)
    # list of cross validation results. 
    return(cv_result_list)


# In[7]:


#def main():
#read data
drug_data = read_drug(drug_path = drug_path)
m_f1, m_f2, m_f3,_  = file_list
m_f1 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f1 + '.csv', index_col = 0)
m_f2 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f2 + '.csv', index_col = 0)
m_f3 = pd.read_csv(path + 'CCLE_mutations_bool_' + m_f3 + '.csv', index_col = 0)
print('#### reading files completed')

#merge mutation files
m_file = merge_mt(merge_mt(m_f1, m_f2, drug_id=drug_data.index), m_f3, 
                  drug_id= drug_data.index,gene_filter=False)
#m_file = m_f1.reindex(set.intersection(set(m_f1.index), set(drug_data.index)))
print('#### merge mutation files completed')
print(m_file.shape)
#filter most mutated genes. 
w_m_file = merge_mt(merge_mt(m_f1, m_f2, drug_id=drug_data.index, gene_filter=True), m_f3, 
                  drug_id= drug_data.index,gene_filter = True)

#w_m_file = m_f1
#m_file = m_f1
gene_bool = filter_genes(w_m_file, thresh=0.08)
#print(gene_bool)
m_file = m_file.loc[:,gene_bool]
print(m_file.shape)

# get rid of drugs with too many missing values.
thresh  = int(m_file.shape[0]*(1-0.1)) ## at least ## number of Non-na value in each drug
drug_id_list = drug_data.loc[m_file.index,:].dropna(axis = 1, thresh = thresh).columns
#build model for each drug.
pre_matrix = pd.Series(index = m_file.index)

''' multiprocessing the command '''
n_jobs = 100
jobs_list = tools.split_task(len(drug_id_list), n_jobs)
drug_mapping_list = list(zip(drug_id_list, [np.where(drug_data.columns == drug_id)
for index_array in jobs_list:
    work_list = []
    for job in index_array:
        drug_name, drug_index = drug_mapping_list[job][0], int(drug_mapping_list[job][1][0])
        p = multiprocessing.Process(target = drug_model, args = (drug_index, m_file, drug_data))
        work_list.append(p)
    #start the programs.
    for job in work_list: job.start()
    #end the programs.
    for job in work_list: job.join()

# for index, (drug_name, drug_index) in enumerate(list(zip(drug_id_list, [np.where(drug_data.columns == drug_id) for drug_id in drug_id_list]))):
#     if index == 3:
#         result = drug_model(comp_index = int(drug_index[0]), m_file = m_file, drug_data=drug_data,
#                        out_path= out_path)
#     #pre_matrix = pd.concat([pre_matrix, result], axis = 1)
    

    
    


# In[ ]:


if __name__ == '__main__':
    main()

