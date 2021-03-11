import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
 

def scaling_x_data(X_train,X_test):
    """apply StandardScaler() to X_train data"""
    
    """* preventing information about the distribution of the test set leaking into the model
       * scaler should be used for after spiliting train and test data
       * Never "fit_transform" on X_test! 
       * value range : 0 to 1"""

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train,  X_test 

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')