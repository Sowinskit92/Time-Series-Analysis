#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 10:54:51 2024

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

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(os.getcwd() + "/Index2018.csv")
df["date"] = pd.to_datetime(df["date"], format = "%d/%m/%Y")
df.set_index("date", inplace = True)
df = df.asfreq("b")

df["spx"] = df["spx"].ffill() # front fill
df["ftse"] = df["ftse"].bfill() # back fill
df["dax"] = df["dax"].fillna(value = df["dax"].mean()) # single value

size = int(len(df.index)*0.8) # sets size for the training/testing dataframes 

df_train = df.iloc[:size]
df_test = df.iloc[size:]


df_train["Market value"] = df_train["ftse"]

df_train["Returns"] = df_train["Market value"].pct_change(1).mul(100)

model_AR7 = ARIMA(df_train["Market value"], order = (7, 0, 0)) # creates model
results_AR7 = model_AR7.fit() # fits model

df_train["res_price"] = results_AR7.resid

# mean of the residuals is close to 0, which suggests the model performs well
print(df_train["res_price"].mean())

# but the variance is very high, ie the errors aren't consistently small, some are huge 
print(df_train["res_price"].var())

df_test = sts.adfuller(df_train["res_price"])
print(df_test)


# plotting ACF to see if the coefficients of the lagged values are related to each other
sgt.plot_acf(df_train["res_price"], zero = False, lags = 40)
plt.title("ACF of Residuals of Prices", size = 24)
plt.ylim(-0.1, 0.1)
plt.show()

# residual prices are mostly within the blue area, so not significant, but some of them 
# are significant, which hints that there is a better model we should be using


# plotting the residuals show no obvious trebd
df_train["res_price"].plot()
plt.title("Residuals of prices", size = 24)
plt.ylim(-500, 500)
plt.show()

# repeating the above steps for returns
model_ret_ar6 = ARIMA(df_train["Returns"], order = (6, 0, 0))
results_ret_ar6 = model_ret_ar6.fit()

df_train["res_returns"] = results_ret_ar6.resid.fillna(0)

# mean is ~0
print(df_train["res_returns"].mean())

# variance is tiny. These are much closer to what we'd expect for white noise data
print(df_train["res_returns"].var())

df_test = sts.adfuller(df_train["res_returns"])
print(df_test)

# plotting ACF to see if the coefficients of the lagged values are related to each other
sgt.plot_acf(df_train["res_returns"], zero = False, lags = 40)
plt.title("ACF of Residuals of Returns", size = 24)
plt.ylim(-0.1, 0.1)
plt.show()

# again, most but not all of the ACFs are insignificant. So there is a better model we can use


