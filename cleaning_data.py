import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.utils_modeling import scaling_x_data


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
print(X_test_scaled.shape)