"""

Bacis classifiers baseline, logreg

__author__: Tatiana Gabruseva
 
"""

import itertools
import os
import pickle
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  SGDClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import plot_confusion_matrix


from create_labels import (apgar5_binary_labels, create_labels,
                            pH_apgar5_binary_labels, pH_binary_labels)

DATA_DIR = '../../data/database/database/signals'
META_FILE = f'../../data/folds_old/df_test{test_num}_folds.csv'
RESULTS_DIR = '../../output/pics'
FEATURES_FILE = '../../data/EHR_FHR_features_20200905.csv'

# clinical factors
clinical_columns = [
    'Gest. weeks', 
    'Sex', # sometimes we know before / sometimes not
    'Age', 
    'Gravidity', 
    'Parity',
    'Diabetes', 
    'Hypertension', 
    'Preeclampsia', 
    'Liq.', 
    'Pyrexia',   
    'Meconium',
    'Presentation', # the position of the baby, important
    'Induced', 
    'I.stage', 
    'NoProgress', 
    'CK/KP',
    'II.stage'
]

# params 
params = {    
    'penalty': 'l2', 
    'tol': 0.0001, # tolerance for stopping criteria
    'C': 0.001, 
    'class_weight': 'balanced', 
    'random_state': 0, 
    'solver':'lbfgs', 
    'max_iter': 100, 
    'multi_class':'ovr', 
    'verbose': 0, 
    'warm_start': False,    
    'l1_ratio': None,    
}

selected_features = [



]

def merge_df(df_train: pd.DataFrame) -> pd.DataFrame:
    df_feat = pd.read_csv(FEATURES_FILE)
    exclude = ['ph', 'bdecf', 'pco2', 'be', 'apgar1', 'apgar5', 'nicu days',
        'seizures', 'hie', 'intubation', 'main diag.', 'other diag.',
        'gest. weeks', 'weight(g)', 'sex', 'age', 'gravidity', 'parity',
        'diabetes', 'hypertension', 'preeclampsia', 'liq.', 'pyrexia',
        'meconium', 'presentation', 'induced', 'i.stage', 'noprogress', 'ck/kp',
        'ii.stage', 'deliv. type', 'dbid', 'rec. type', 'pos. ii.st.',
        'sig2birth']
    df_feat = df_feat.drop(exclude, axis = 1)        
    df = pd.merge(df_train, df_feat, how='inner', on='patient', left_on=None, right_on=None,
            left_index=False, right_index=False, sort=True,
            suffixes=('_x', '_y'), copy=True, indicator=False,
            validate=None)
    
    return df


def scale_features(df: pd.DataFrame, scale_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scale features using standard scaler
    """
    if scale_columns == None:
        scale_columns = [
            'Gest. weeks', 'Age', 'Gravidity', 'Parity', 'I.stage', 'II.stage', 'pH', 'mad', 'spkt_welch_density_100', 'median', 
            'autocorrelation_50', 'ptp', 'var', 'max', 'complexity', 'spkt_welch_density_1', 'ratio_unique_values', 'change_rate', 
            'mean_peak_prominence', 'wavelet_square_1_mean', 'wavelet_square_0_var', 'moment_3', 'wavelet_square_1_mean', 
            'abs_energy', 'wavelet_square_3_max', 'wavelet_square_6_max', 'wavelet_square_0_var',  'mean_abs_change', 
            'wavelet_square_9_max', 'wavelet_square_5_mean', 'wavelet_square_10_max', 'fft_2_real', 'binned_entropy_10', 
            'wavelet_square_6_mean', 'wavelet_square_4_mean', 'wavelet_square_2_mean', 'wavelet_square_9_mean', 'wavelet_square_7_mean', 
            'wavelet_square_8_max', 'fft_1_imag', 'wavelet_square_8_mean', 'time_rev_asym_stat_100', 'fft_1_real', 'wavelet_square_7_max',
            'wavelet_square_2_max', 'wavelet_square_5_max', 'average_change', 'wavelet_square_4_max', 'fft_3_real', 'fhr_detrended_var',                         
            'uc_mean_prominances', 'uc_mean_peak_hgt', 'uc_per10mins', 'uc_mean_dur', 'fhr_accel_ratio_counts', 'fhr_mean_accel_prominances', 
            'fhr_mean_accel_peak_hgt', 'fhr_accel_per10mins', 'fhr_mean_accel_dur', 'fhr_decel_ratio_counts', 'fhr_mean_decel_prominances',
            'fhr_mean_decel_peak_hgt', 'fhr_decel_per10mins', 'fhr_mean_decel_dur', 'fhr_vlf_psd', 'fhr_lf_psd', 'fhr_mf_psd', 
            'fhr_hf_psd', 'fhr_freq_ratio', 
            ]
    df['Sex'] = df['Sex'].replace(2, 0)
    df[scale_columns] = StandardScaler().fit_transform(df[scale_columns])

    return df


def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df['2stage'] = le.fit_transform(df['II.stage'].values)
    print(df['2stage'].unique())

    return df


def reset_state(seed_num: int = 1234):
    np.random.seed(seed_num)
    import random
    random.seed(seed_num)
    os.environ["PYTHONHASHSEED"] = "0"


def get_fold_data(df_train: pd.DataFrame, used_columns: list, target_column: str, fold: int):
    """
    Get features and labels for a single fold for train and validation    
    """
    x_train = df_train[used_columns][df_train['fold'] != fold].values
    y_train = df_train[target_column][df_train['fold'] != fold].values
    x_valid = df_train[used_columns][df_train['fold'] == fold].values
    y_valid = df_train[target_column][df_train['fold'] == fold].values
    
    return x_train, y_train, x_valid, y_valid
    

def train_fold(df_train: pd.DataFrame, used_columns: list, target_column: str, 
               fold: int, clf, save_name: Optional[str] = None)-> dict:
    """
    Train single fold, classification with sklearn clfs, calculate permutation importances and metrics
    Save classifier (optionally)   

    Args: 
        df_train: (pd.DataFrame) patients meta data    
        used_columns: (list) list of features columns 
        target_column: (str) name of target column 
        fold: (int) number of fold
        clf: initiated classifier object
        save_name: Optional[str] If not None saves clf to txt to the save_name path

    Output: 
        results: (Dict) Dictionary with metrics and trained classifier
    """
    x_train, y_train, x_valid, y_valid = get_fold_data(df_train, used_columns, target_column, fold)    
    # classifier    
    clf.fit(x_train, y_train)    
    # permutation importance
    imp_train = permutation_importance(clf, x_train, y_train, n_repeats=5,
                                random_state=1234, n_jobs=1)
    imp_valid = permutation_importance(clf, x_valid, y_valid, n_repeats=5,
                                random_state=1234, n_jobs=1)    
    # preds binarised at 0.5
    y_pred = clf.predict(x_valid) 
    results = {}
    # metrics
    results['auc'] = roc_auc_score(y_valid, y_pred)
    results['f1'] = f1_score(y_valid, y_pred)
    results['recall'] = recall_score(y_valid, y_pred)
    results['precision'] = precision_score(y_valid, y_pred)
    results['conf_matrix'] = confusion_matrix(y_valid, y_pred)
    results['y_pred'] = y_pred    
    results['clf'] = clf
    results['per_imp_val'] = imp_valid
    results['per_imp_train'] = imp_train    
    # save model to file
    if save_name:
        pickle.dump(clf, open(f'{save_name}_{fold}.txt', 'wb'))
    
    return results


def train_folds(df_train: pd.DataFrame, used_columns: list, target_column: str, clf, save_name: Optional[str] = None) -> dict:
    """
    Train cross-validated folds, classification with sklearn clfs, calculate mean permutation importances and metrics
    Get the list of clfs

    Args: 
        df_train: (pd.DataFrame) patients meta data    
        used_columns: (list) list of features columns 
        target_column: (str) name of target column 
        clf: initiated classifier object

    Output: 
        results: (Dict) Dictionary with metrics and trained classifiers
    """
    auc, f1, precision, recall = [], [], [], []
    clfs, imp_mean, imp_std = [], [], []
    for fold in range(5):
        results = train_fold(df_train, used_columns, target_column, fold, clf)
        auc.append(results['auc'])
        f1.append(results['f1'])
        recall.append(results['recall'])
        precision.append(results['precision'])         
        imp = results['per_imp_val']        
        imp_mean.append(imp.importances_mean)
        imp_std.append(imp.importances_std)   
        clfs.append(results['clf']) 
        
    results = {}
    # metrics
    results['mean_auc'] = np.mean(auc) 
    results['mean_f1'] = np.mean(f1)
    results['mean_recall'] = np.mean(recall)
    results['mean_precision'] = np.mean(precision)
    results['imp_mean'] = np.mean(imp_mean, axis = 1)
    results['imp_std'] = np.mean(imp_std, axis = 1)  
    results['clfs'] = clfs
    
    return results 
    

def plot_cm(df_train: pd.DataFrame, used_columns: list, target_column: str, fold: int, clf):
    x_train, y_train, x_valid, y_valid = get_fold_data(df_train, used_columns, target_column, fold)
    class_names = ['norm', 'pathol']    
    disp = plot_confusion_matrix(clf, x_valid, y_valid,
                                display_labels=class_names,
                                cmap=plt.cm.Blues)


def one_test_fold(test_num: int, features: List[str], target_column: str, clf):
    """
    Select one test fold and perform CV training and precition
    """
    reset_state(1234)
    df = pd.read_csv(f'../../data/folds_old/df_test{test_num}_folds.csv')
    # prepare labels and features
    df_train = create_labels(df)
    #print_class_distr(df_train)
    df = merge_df(df_train)
    # remove intermediate
    df_train = df[df[target_column] != -1]
    df_train = scale_features(df_train)
    df_train = label_encoding(df_train)
    # train CV folds
    results = train_folds(df_train, features, target_column, clf) 
    print(results)

    fold = 0
    plot_cm(df_train, features, target_column, fold, clf = results['clfs'][fold])

    # load clfs


if __name__ == "__main__":
    clf = LogisticRegression(**params)  
    # test number
    test_num = 0
    