#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:11:30 2024

@author: timsowinski
"""
import statsmodels
import pandas as pd
import os
import scipy.stats
import pylab
import matplotlib.pyplot as plt
import numpy as np
import sys
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from scipy.stats.distributions import chi2
from statsmodels.tsa.arima.model import ARIMA

def LLR_test(model_1, model_2, DF = 1):
    
    # DF = degrees of freedom, usually we only do this test for models with a difference of degrees of 
    # freedome of 1 (eg AR2 and AR3), so this is why the default argument is 1
    L1 = model_1.fit().llf
    L2 = model_2.fit().llf
    LR = 2*(L2 - L1) # test statistic for this test
    
    p = chi2.sf(LR, DF).round(5)
    
    name1 = f"{model_1=}".split("=")[0]
    name2 = f"{model_2=}".split("=")[0]
    
    print(f"p-value of LLR test between {name1} & {name2} is: {p}")
    return p

df = pd.read_csv(os.getcwd() + "/Index2018.csv")
df["date"] = pd.to_datetime(df["date"], format = "%d/%m/%Y")
df.set_index("date", inplace = True)
df = df.asfreq("b")

df["spx"] = df["spx"].ffill() # front fill
df["ftse"] = df["ftse"].bfill() # back fill
df["dax"] = df["dax"].fillna(value = df["dax"].mean()) # single value

df["Market value"] = df["ftse"]

size = int(len(df.index)*0.8) # sets size for the training/testing dataframes 

#df_train = df
df_train = df.iloc[:size]
df_test = df.iloc[size:]


df_train["Returns"] = df_train["Market value"].pct_change(1).mul(100)

print(df_train)

"""
Creating AR1 model to predict returns
"""
model_ret_ar1 = ARIMA(df_train["Returns"], order = (1, 0, 0))
results_ret_ar1 = model_ret_ar1.fit()
results_ret_ar1_summary = results_ret_ar1.summary()
#print(results_ret_ar1_summary)

# P value of the constant is over 0.05, so isn't significantly different from 0

# doing the same for AR2 and running the LLR test we find
model_ret_ar2 = ARIMA(df_train["Returns"], order = (2, 0, 0))
results_ret_ar2 = model_ret_ar2.fit()
results_ret_ar2_summary = results_ret_ar2.summary()
#print(results_ret_ar2_summary)
LLR_test(model_ret_ar1, model_ret_ar2)

# p-value is less than 1% which supports the claim that AR2 is a better model than AR1

# doing the same for AR3 and running the LLR test we find
model_ret_ar3 = ARIMA(df_train["Returns"], order = (3, 0, 0))
results_ret_ar3 = model_ret_ar3.fit()
results_ret_ar3_summary = results_ret_ar3.summary()
#print(results_ret_ar3_summary)
LLR_test(model_ret_ar2, model_ret_ar3)

# p value of highest lags are small, and LLR test is ~0 so this is a better model than AR2

# repeating the above steps until either the LLR test fails, the Information Criteria goes up,
# or the new coefficient is insignificant will show that AR6 is the best model (AR7 is worse 
# than AR6 but AR6 is better than all the others)

model_ret_ar6 = ARIMA(df_train["Returns"], order = (6, 0, 0))
results_ret_ar6 = model_ret_ar6.fit()
results_ret_ar6_summary = results_ret_ar6.summary()
print(results_ret_ar6_summary)

model_ret_ar7 = ARIMA(df_train["Returns"], order = (7, 0, 0))
results_ret_ar7 = model_ret_ar7.fit()
results_ret_ar7_summary = results_ret_ar7.summary()
print(results_ret_ar7_summary)
LLR_test(model_ret_ar2, model_ret_ar3)

LLR_test(model_ret_ar6, model_ret_ar7)

# AR7 fails across basically all the criteria




