
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def return_categorical_df(df):
    """data=df type"""
    
    # choose only object type variables
    temp_df = df.select_dtypes(include=[object])
    # save object type variables in list
    temp_df_col = temp_df.select_dtypes(include=[object]).columns.tolist()
    
    # check null
    null_series = temp_df.isnull().any()
    # most frequest values
    freq = temp_df[temp_df_col].mode().T
    freq.rename(columns = {0:'frequent'}, inplace = True)
    # count values and unique values
    df_cat = temp_df.agg(['count','nunique']).T
    
    # concat DFs to a df
    df_cat =  pd.concat([df_cat, null_series,freq], axis=1)
    # rename colmuns
    df_cat = df_cat.rename(columns={"count": "Count", "nunique": "Unique_value",0:"Missing_value","frequent":"Most_frequent"})
    return df_cat




def calc_vif(X):

    """Calculating VIF"""
    """X = df type"""
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

