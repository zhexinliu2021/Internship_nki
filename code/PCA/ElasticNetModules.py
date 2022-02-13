# Modules for the ElasticNet model
# and model training and validation.
# Date: 11/02/2022


import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings
import tools


def EN_cv_in(x_train, y_train, fold, alpha, l1_ratio):
    """
    INPUT: training set. (each alpha, l1_ratio parameter)
    OUTPUT: averge spearman correlation score of 10 results
    FUNCTION: conduct 10 cross validation on the 80 % of resampled dataset
              to determine the best l1_ratio and alpha pair.
    """
    # alpha = p1, l1_ratio = p2
    kf_in = KFold(n_splits=fold)
    cv_perf = np.array([])
    for random_, (train_in, test_in) in enumerate(kf_in.split(x_train)):
        x_train_in, x_test_in, y_train_in, y_test_in = x_train[train_in], x_train[test_in], y_train[train_in], y_train[
            test_in]

        # build model with each combination of parameters.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regr = ElasticNet(random_state=random_, alpha=alpha, l1_ratio=l1_ratio,
                              fit_intercept=False, max_iter=2000)
            regr.fit(x_train_in, y_train_in)

        y_pre = regr.predict(x_test_in)
        # perforamance matrix of the model is the Pearson's corrlation coeeficient r.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = pd.Series(y_pre).corr(pd.Series(y_test_in), method='spearman')
            cv_perf = np.append(cv_perf, [r])
            del regr
    return (np.mean(cv_perf))


def out_loop(x, y_total, cl_id, out_cv_fold=5):
    ''' 5 fold cv (outter-loop)
    # x: (#_(80%)C, #_genes); y: (#_(80%)C,)
    FUNCTION: conduct 5-fold cross validation on the resampled data. In each iteration, conduct
              10 fold CV on the training set (EN_cv_in function) to do parameter search and select the l1
              ratio and alpha with highest validation score on the 10th fold testing set
    '''
    kf = KFold(n_splits=out_cv_fold)
    pre_y_array = dict(list(zip(cl_id, [[] for _ in range(len(cl_id))])))
    per_test_list = []
    for train, test in kf.split(x):
        x_train, x_test, y_train, y_test = x.values[train], x.values[test], y_total.values[train], y_total.values[test]
        train_label, test_label = x.index[train], x.index[test]
        # 10-fold in x_train (inner-loop).
        # each pair of alpha and l1_ratio, do cross training to sellect best pair.

        l1_ratio_list = np.linspace(start=0.2, stop=1.0, num=5)  # 10 values
        #l1_ratio_list  = np.linspace(start=1e-3, stop=0.2, num=5)
        #alpha_list = np.array([math.exp(i) for i in np.arange(-6, 5, 0.8)])  # 250 values
        alpha_list =  np.array([math.exp(i) for i in np.arange(-15, 5, 2)])   #  values
        para_matrix = {(l1_ratio, alpha): 0 for l1_ratio in l1_ratio_list for alpha in alpha_list}

        for (l1_ratio, alpha) in para_matrix:
            # do ten fold cv to sellect best parameter pairs.
            ave_per = EN_cv_in(x_train=x_train, y_train=y_train, fold=8, alpha=alpha, l1_ratio=l1_ratio)
            para_matrix[(l1_ratio, alpha)] = ave_per
        print(para_matrix, flush=True)

        # return the best alpha-l1-ratio pair.
        op_l1_ratio, op_alpha = pd.Series(para_matrix).idxmax()
        print('the best validation performance is: ', para_matrix[(op_l1_ratio, op_alpha)], flush=True)

        # predict on the 5th-fold set (testing set)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            regr = ElasticNet(random_state=0, alpha=op_alpha, l1_ratio=l1_ratio,
                              fit_intercept=False, max_iter= 2000)
            regr.fit(x_train, y_train)
            y_pre = regr.predict(x_test)

        if len(np.unique(y_pre)) == 1: print('constant prediction.', flush=True)
        del regr
        # add results to the storage dictionary.
        for index, value in zip(test_label, y_pre): pre_y_array[index].append(value)
        per_test = pd.Series(y_pre).corr(pd.Series(y_test), method='spearman')
        per_test_list.append(per_test)
        print('performance on the test set is:', per_test, flush=True)
        plt.scatter(y_test, y_pre)
        plt.xlabel('test set')
        plt.ylabel('prediction')
        plt.show()
        # break
    # add the prediction to the list.
    pre_y_array = pd.Series(pre_y_array).reindex(cl_id)

    return (pre_y_array, per_test_list)


def drug_model(m_file, target, out_path, norm_bool = True):
    """
    INPUT: m_file mutation matrix with unpreprocessed binery values.
    OUTPUT: mean prediction values of 10 iterations (TYPE: pd.Sereis).
    FUNCTION: conduct normalization + 10 times of 5-fold crossvalidation
    """
    # eliminate cell lines with nan drug sensitivity data.
    drug_name = target.name
    Y = target.dropna()
    m_file_notna = m_file.loc[Y.index, :]
    print(f'processing drug: {drug_name}')
    # normalize the each col to have ~0 mean and ~1 sd.
    # old normalization method:
    # norm_m_file = (m_file_notna - m_file_notna.mean(axis = 0)) / m_file_notna.std(axis = 0)

    if norm_bool:
        norm_m_file = tools.scal_matrix(m_file_notna)
        print('### normalize with mean=0, std = 1.', flush=True)
    elif not norm_bool:
        norm_m_file = m_file_notna
        print('### using raw data to fit model', flush=True)

    norm_m_file.dropna(axis=1, inplace=True)
    print('number of features selected {}'.format(norm_m_file.shape[1]), flush=True)
    print(f'Dimension of the final matrix: {norm_m_file.shape}')
    X, Y_total = norm_m_file, Y

    ### EN model
    ## bootstraping (80% of data) for 10 times.
    ## each time with 80% data for training and remaining 20% for
    ## testing (5 fold CV).
    cl_id = norm_m_file.index
    Y_pre_array = pd.Series(dict(list(zip(cl_id, [[] for i in range(cl_id.shape[0])]))),
                            index=cl_id, name=drug_name)
    cv_result_list = []
    for i in range(10):
        print('iteration {} begains'.format(i), flush=True)
        # resample 80% of data
        x, y_total = resample(X, Y_total, replace=False, n_samples=int(X.shape[0] * 0.95), random_state=i)
        #print(X.isna().sum().sum(), flush=True)
        #print(y_total.isna().sum().sum(), flush=True)

        # outter loop with 5 fold cv.
        prediction, cv_list = out_loop(x=x, y_total=y_total, cl_id=cl_id, out_cv_fold=5)
        cv_result_list.extend(cv_list)
        print('iter {}, outter loop finished. '.format(i), flush=True)

        for i in prediction.index: Y_pre_array[i].extend(prediction[i])

    ##calculate the mean of 10 iterations.
    # Y_pre_array = Y_pre_array.map(np.mean)
    tools.write_file(Y_pre_array, 'comp_{}.csv'.format(drug_name), out_path)
    # list of cross validation results.
    return (cv_result_list)