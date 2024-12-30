#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 09:24:20 2024

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

"""
From now on, each time there's a new model introduced, a similar format will be undertaken:
    1. Go over the maths
    2. Start coding
    3. Create simple versions of the model
    4. Fit model for the Returns dataset and find coefficients/LLR tests
    5. Look at the residuals
    6. Find an appropriate number of lags for the model
    7. Will then check if models are good at predicting non-stationary data (eg. Prices rather than returns)
"""

def LLR_test(model_1, model_2, DF = 1):
    
    # DF = degrees of freedom, usually we only do this test for models with a difference of degrees of 
    # freedome of 1 (eg AR2 and AR3), so this is why the default argument is 1
    L1 = model_1.fit().llf
    L2 = model_2.fit().llf
    LR = 2*(L2 - L1) # test statistic for this test
    
    p = chi2.sf(LR, DF).round(5)
    
    name1 = f"{model_1=}".split("=")[0]
    name2 = f"{model_2=}".split("=")[0]
    
    print(f"\np-value of LLR test between {name1} & {name2} is: {p}")
    return p


def ARMA_model(data, AR = 0, MA = 0, params = False, resid = False):
    # I've put this into its own function as it's done so often in the course
    # creating MA1 model. Note order = (AR, MA, Integration)
    model_ARMA = ARIMA(data, order = (AR, 0, MA))
    results_ARMA = model_ARMA.fit()
    results_ARMA_resid = results_ARMA.resid
    results_ARMA_summary = results_ARMA.summary()
    print(results_ARMA_summary)
    print(results_ARMA.params)
    
    if params == True:
        return model_ARMA, results_ARMA.params
    
    elif resid == True:
        return model_ARMA, results_ARMA_resid
    
    else:
        return model_ARMA

"""Setting up data"""
raw_csv_data = pd.read_csv("Index2018.csv")
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace = True)
df_comp = df_comp.asfreq("b")
df_comp = df_comp.ffill()

df_comp["Market value"] = df_comp.ftse

del df_comp["spx"], df_comp["dax"], df_comp["ftse"], df_comp["nikkei"]

size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

df["returns"] = df["Market value"].pct_change(1).mul(100)

"""Plotting ACF for the returns column"""

sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title("ACF of returns", size = 20)
plt.ylim(-0.1, 0.1)
plt.show()

"""
# will look at 8 lags as these are all mostly significant. There are other significant ones
# more lags ago but, as we expect the impact of lags to be smaller the further back in time
# they are, we can disregard these

# creating MA1 model. Note order = (AR, MA, Integration)
MA1 = ARMA_model(df.returns[1:], AR = 0, MA = 1)

# Fitting higher lag MA models
MA2 = ARMA_model(df.returns[1:], AR = 0, MA = 2)
LLR_test(MA1, MA2) # p-value is very close to 0, so MA2 makes much better predictions than MA1 

# repeating until MA7 we find...
MA6 = ARMA_model(df.returns[1:], AR = 0, MA = 6)
MA7 = ARMA_model(df.returns[1:], AR = 0, MA = 7)
LLR_test(MA6, MA7)
# MA7 fails the LLR test and returns a non-significant coefficient
# however, if you look at the ACF plot, the 7th one is insignificant, but the eigth one is
# so we should try this
"""
MA8, MA8_resid = ARMA_model(df.returns[1:], AR = 0, MA = 8, resid = True)
# we're doing it between MA6 and MA8 because MA7 was a worse model than MA6, so we want to know if 
# we should use MA6 or MA8
"""
LLR_test(MA6, MA8, DF = 2) # DF needs to be 2 as there's two degrees of freedom difference between the models
"""
# MA8 is the better model due to the significant coefficient and small p value

df["res_ret_MA8"] = MA8_resid

print(df["res_ret_MA8"].mean())
print(df["res_ret_MA8"].var())

#df["res_ret_MA8"][1:].plot(figsize = (20, 5))
df["res_ret_MA8"][1:].plot()
plt.title("Residuals of returns", size = 20)
plt.show()

# checking the residuals to see if they resemble white noise (ie stationary, but also with no significant ACFs)
print(sts.adfuller(df["res_ret_MA8"][1:]))
# p value is 0, so the data is stationary

sgt.plot_acf(df["res_ret_MA8"][1:], zero = False, lags = 40)
plt.title("ACF of Residuals for Returns", size = 20)
plt.ylim(-0.05, 0.05)
plt.show()

# we can see that most of the ACF coefficients aren't significant 
# some are but again they're all really far back (>18 lags), so shouldn't have a big effect on the model

"""MA Models with Normalised Data"""

bench_ret = df.returns.iloc[1] # creates benchmark prices 
df["norm_ret"] = df.returns.div(bench_ret).mul(100)

sgt.plot_acf(df["norm_ret"][1:], zero = False, lags = 40)
plt.title("ACF of Normalised Returns", size = 20)
plt.ylim(-0.1, 0.1)
plt.show()

# from looking at the ACF plot, it looks like 8 lags is the best model to use to begin with
MA8_norm, MA8_norm_resid = ARMA_model(df["norm_ret"], MA = 8, resid = True)

# Note: If you compare these outputs to the non-normalised returns, you find they're identical

MA8_norm_resid[1:].plot()
plt.title("Residuals of returns", size = 20)
plt.show()

sgt.plot_acf(MA8_norm_resid[1:], zero = False, lags = 40)
plt.title("ACF of Normalised Returns", size = 20)
plt.ylim(-0.1, 0.1)
plt.show()

# ACF resembles white noise
print(sts.adfuller(MA8_norm_resid.bfill()))

"""Moving Average model on non-stationary data (ie prices)"""
sgt.plot_acf(df["Market value"], zero = False, lags = 40)
plt.title("ACF for Prices", size = 20)
plt.show() # appears all 40 lags are significant, suggesting any higher lag model would be better than others
# this suggests that moving average models won't work too well for this but lets look anyway

MA1_prices = ARMA_model(df["Market value"], MA = 1)
# coefficient of MA.L1 is very close to 1 (ie it's saying to keep essentially all of the error term
# from the previous lag)

"""Conclusion: MA models don't perform well for non-stationary data. While they take into account the error
terms, they still don't do a good job at predicting non-stationary data because in order to do a good
job, the model needs to know the previous value (which the AR model does, but then this doesn't take
                                                 into account the error)
So, some combination of the AR and MA models should be better for predicting non-stationary data
"""

