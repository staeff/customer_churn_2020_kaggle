import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("data/train.csv")

# indices of churn no ourlier
df_outlier_index = df.loc[(df['churn'] == "no" ) & (df["number_customer_service_calls"]>=4)].index
# remove outliers
df.drop(index = df_outlier_index,inplace =True)


#Create a label encoder object
le = LabelEncoder()
# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        if len(list(df[col].unique())) <= 2:
            le.fit(df[col])
            df[col] = le.transform(df[col])
            le_count += 1
#print('{} columns were label encoded.'.format(le_count))


# Dummy variables
# drop_first=True = show less columns and avoid multicolinility
df=pd.get_dummies(df,drop_first=True )

# Feature engineering
# create a new bin from "account_length". Rank 1-10
df["account_length_group"] = pd.qcut(df['account_length'].rank(method = 'first'),q=5,labels=range(5,0,-1))

# drop "account_length"
df =df.drop(columns=["account_length"])
print(df.shape)

# Save cleaned df
df.to_csv("data/test_cleaned_df.csv",index=False)