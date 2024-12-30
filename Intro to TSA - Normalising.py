#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:01:06 2024

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

# we'll pick the first element to normalise the data to
benchmark = df_train["Market value"].iloc[0]
df_train["Normalised"] = df_train["Market value"].div(benchmark).mul(100)

# checking the Normalised data to see if it's stationary
df_test = sts.adfuller(df_train["Normalised"])
print(df_test)
# pvalue is ~0.3 so the data is likely non-stationary, therefore we can't use an AR model
# for the normalised prices

# BUT - we could try to normalise the returns which were stationary
# they're also good as they show the profitability of investments 
benchmark = df_train["Returns"].iloc[1x] # should be market value not returns as it's just a normalisation (so you could do it to any number you want)
df_train["Normalised"] = df_train["Returns"].div(benchmark).mul(100)
print(df_train)
# checking the Normalised data to see if it's stationary
# df_test = sts.adfuller(df_train["Normalised"])
#print(df_test)

# df_test gives a test statistic (first value of ~ -12 which is much less than the ~ -3 of the 1% critical value)
# the pvalue (second value) is also very small

"""Got to 44:00 mins in the video https://www.youtube.com/watch?v=hprO9_VtKso&list=PLtIY5kwXKny91_IbkqcIXuv6t1prQwFhO&index=8"""

# it's always good to check if normalising the data has an effect on the model coefficients
# sometimes it will and sometimes it won't
