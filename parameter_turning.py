import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


import eli5
from eli5.sklearn import PermutationImportance

from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

import pickle
from datetime import datetime
import timer 


from utils.utils_modeling import scaling_x_data


pd.set_option('display.max_columns', None) 
df=pd.read_csv("data/test_cleaned_df.csv")


X = df[['account_length_group', 'international_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'number_customer_service_calls',
       'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO', 'state_CT',
       'state_DC', 'state_DE', 'state_FL', 'state_GA', 'state_HI', 'state_IA',
       'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY', 'state_LA',
       'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN', 'state_MO',
       'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE', 'state_NH',
       'state_NJ', 'state_NM', 'state_NV', 'state_NY', 'state_OH', 'state_OK',
       'state_OR', 'state_PA', 'state_RI', 'state_SC', 'state_SD', 'state_TN',
       'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WA', 'state_WI',
       'state_WV', 'state_WY', 'area_code_area_code_415',
       'area_code_area_code_510']]
y= df[['churn']]

# split test_cleaned_df.csv into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# scaling X_train, X_test
# scaler should be used for after spiliting train and test data
X_train_scaled,  X_test_scaled = scaling_x_data(X_train,X_test)




XGB model and its parameters
XGB =XGBClassifier()
pprint(XGB.get_params())

# kfold 10
kf = KFold(n_splits=10, random_state=42, shuffle=False)

params_xgb = {#'n_estimators': [100], this is default
               'max_depth': [6,8,10,"None"],
               #'validate_parameters': [True], this is default
               'min_child_weight': [1,2,3,"None"],
               'gamma':[0, 0.5,"None"],
               'learning_rate':[0.05,0.1,0.3,0,4,"None"],
               'colsample_bytree':[1,0.5,"None"]
}

# Scoring ="f1_macro"
grid_no_up = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='f1_macro').fit(X_train_scaled,y_train)
print(grid_no_up.best_score_)
print(grid_no_up.best_params_)
print(f1_score(y_test, grid_no_up.predict(X_test_scaled),average='macro'))


# scoring = "arrucary"
grid_no_up = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='accuracy').fit(X_train_scaled, y_train)

print(grid_no_up.best_score_)
print(grid_no_up.best_params_)


