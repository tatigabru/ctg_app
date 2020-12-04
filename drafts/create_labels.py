"""

Creates labels based on pH and Apgar5

__author__: Tatiana Gabruseva
 
"""
import os
import sys
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def pH_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        pH >= 7.15 - normal, label = 0
        pH >= 7.05 and  < 7.15 - intermediate, label = 2
        pH < 7.05 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1   
    df_train[target_col][df_train['pH'] >= 7.2] = 0 # 375       
    df_train[target_col][(df_train['pH'] < 7.2)&(df_train['pH'] >= 7.15)] = 1  # 72  
    df_train[target_col][(df_train['pH'] < 7.15)&(df_train['pH'] >= 7.1)] = 2 # 49
    df_train[target_col][(df_train['pH'] < 7.1)&(df_train['pH'] >= 7.0)] = 3  # 36   
    df_train[target_col][df_train['pH'] < 7.0] = 4 # 20
    
    return df_train


# binary labels
def pH_binary_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        pH >= 7.2 - normal, label = 0
        pH >= 7.05 and  < 7.2 - intermediate, label = -1
        pH < 7.05 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1   
    df_train[target_col][df_train['pH'] >= 7.2] = 0 # 375    
    df_train[target_col][df_train['pH'] <= 7.05] = 1 
    
    return df_train


def apgar5_binary_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on Apgar5 values:    
        Apgar5 = 9, 10 - normal, label = 0
        Apgar5 = 6, 7, 8 - imermediate, label = -1
        Apgar5 < 6 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1   
    df_train[target_col][df_train['Apgar5'] > 8] = 0 # 375    
    df_train[target_col][df_train['Apgar5'] < 6] = 1 
    
    return df_train


def pH_apgar5_binary_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        pH >= 7.15 & Apgar5 = 9, 10 - normal, label = 0
        pH > 7.05 and  < 7.15 - intermediate, label = -1
        pH <= 7.05 or Apgar5 < 6 - pathological, label = 1 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1   
    df_train[target_col][(df_train['pH'] >= 7.15)&(df_train['Apgar5'] > 8)] = 0 # 375    
    df_train[target_col][(df_train['pH'] <= 7.05)|(df_train['Apgar5'] < 6)] = 1 
    
    return df_train

    
def argar5_labels(df_train: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Generate label for each signal based on pH values:    
        Apgar5 > 8 - normal, label = 0
        Apgar5 == 6, 7 - intermediate, label = 2
        Apgar5 <= 5 - pathological, label = 3 

    Args: 
        df_train: (pd.DataFrame) patients meta data    

    Output: patient df with labels
    
    We only consider first stage of labour 
    """
    # create target column
    df_train[target_col] = -1  
    df_train[target_col][(df_train['Apgar5'] > 8)] = 0  
    df_train[target_col][(df_train['Apgar5'] == 8)] = 1        
    df_train[target_col][(df_train['Apgar5'] == 6)|(df_train['Apgar5'] == 7)] = 2
    df_train[target_col][(df_train['Apgar5'] <= 5)] = 3    

    return df_train


def create_labels(df: pd.DataFrame) -> pd.DataFrame:    
    """        
    Create labels
    """ 
    df_train = apgar5_binary_labels(df, 'binary_apgar5')
    df_train = pH_binary_labels(df_train, 'binary_ph')
    df_train = pH_apgar5_binary_labels(df_train, 'target_ph_apgar5')
    
    return df_train


def print_class_distr(df: pd.DataFrame) -> None:
    """
    Print distribution of different outputs
    """
    print(df['target_ph_apgar5'].value_counts())
    print(df['binary_apgar5'].value_counts())
    print(df['binary_ph'].value_counts())