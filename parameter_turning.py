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
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

import pickle

from cleaning_data import X_train_scaled, X_test_scaled, y_train,y_test

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
grid_f1 = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='f1_macro').fit(X_train_scaled,y_train)
print(grid_f1_xgb.best_score_)
print(grid_f1_xgb.best_params_)
print(f1_score(y_test, grid_f1_xgb.predict(X_test_scaled),average='macro'))


# scoring = "arrucary"
grid_acc_xgb = GridSearchCV(XGB, param_grid=params_xgb, cv=kf, 
                          scoring='accuracy').fit(X_train_scaled, y_train)

print(grid_acc_xgb.best_score_)
print(grid_acc_xgb.best_params_)



# GBC model and its parameters
GBC =GradientBoostingClassifier()
pprint(GBC.get_params())

params_gbc ={"learning_rate" : [1, 0.5, 0.1],
             "n_estimators" : [50, 100, 200],
             "max_depth": [3,6,10,"None"],
             "min_samples_split": [0.5,1,2],
             "min_samples_leaf":[0.5,1,2],
             #"max_features":list(range(1,X_train.shape[1])),
             }

# Scoring ="f1_macro"
grid_f1_gbc = GridSearchCV(GBC, param_grid=params_gbc, cv=kf, 
                          scoring='f1_macro').fit(X_train_scaled,y_train)
print(grid_f1_gbc.best_score_)
print(grid_f1_gbc.best_params_)
print(f1_score(y_test, grid_f1_gbc.predict(X_test_scaled),average='macro'))

# scoring = "arrucary"
grid_acc_gbc = GridSearchCV(GBC, param_grid=params_gbc, cv=kf, 
                          scoring='accuracy').fit(X_train_scaled, y_train)

# "arrucary"
print(grid_acc_gbc.best_score_)
print(grid_acc_gbc.best_params_)



# Save a model
# set best parameters from grid search
XGB_turned= XGBClassifier(colsample_bytree= 0.5, gamma= 0, learning_rate=0.1, max_depth= 6, min_child_weight= 3)
# save the model to disk
pickle_model = 'model/XGB_turned.sav'
pickle.dump(XGB_turned, open(pickle_model, 'wb'))
