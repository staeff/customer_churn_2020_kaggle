import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.utils_cleaning_data import return_categorical_df 
from utils.utils_cleaning_data import calc_vif

df=pd.read_csv("data/train.csv")

print(df.info())

df_crosstab =pd.crosstab(index=df['state'], columns=df['churn'])
df_crosstab["churn_total"] =df_crosstab["no"].values + df_crosstab["yes"].values
df_crosstab["churn%"] = df_crosstab["yes"]/(df_crosstab["churn_total"])
df_crosstab = df_crosstab.sort_values(by=["churn%"], ascending=False)
print(df_crosstab.head())

print(return_categorical_df(df))

df=pd.read_csv("data/test_cleaned_df.csv")

vif_cols=df[["voice_mail_plan", "number_vmail_messages","total_day_minutes", "total_day_charge","total_eve_minutes", "total_eve_charge",
        "total_night_minutes", "total_night_charge","total_intl_minutes", "total_intl_charge"]]

vif_cols2 = df[['account_length', 'international_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'number_customer_service_calls',
       'area_code_area_code_415', 'area_code_area_code_510']]

vif_cols3 = df[['account_length', 'international_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'number_customer_service_calls',
       'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO',
       'state_CT', 'state_DC', 'state_DE', 'state_FL', 'state_GA', 'state_HI',
       'state_IA', 'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY',
       'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN',
       'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE',
       'state_NH', 'state_NJ', 'state_NM', 'state_NV', 'state_NY', 'state_OH',
       'state_OK', 'state_OR', 'state_PA', 'state_RI', 'state_SC', 'state_SD',
       'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WA',
       'state_WI', 'state_WV', 'state_WY',
       'area_code_area_code_415', 'area_code_area_code_510']]


print(calc_vif(vif_cols))
print(calc_vif(vif_cols2))
print(calc_vif(vif_cols3))