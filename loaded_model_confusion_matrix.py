import pandas as pd
import numpy as np
import pickle
from cleaning_data import X_train_scaled, X_test_scaled, y_train,y_test

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


# load "picked_model" and define as "loaded_model" 
loaded_model = pickle.load(open("model/XGB_turned.sav",'rb'))
# train model with turned hyperparameters
loaded_model.fit(X_train_scaled, y_train)
# return predection
y_pred =loaded_model.predict(X_test_scaled)
result = loaded_model.score(X_test_scaled, y_test)
print("best prediction of XGB")
print(result)


# Confusion matrix
loaded_model_confusion_matrix =confusion_matrix(y_test,y_pred)
class_names = ["No churn", "Churn"]
fig,ax =plot_confusion_matrix(conf_mat = loaded_model_confusion_matrix,colorbar = True,
                             show_absolute=False, show_normed=True,
                             class_names = class_names)
plt.show()

# Classification report 
print(classification_report(y_test,y_pred))