#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:58:28 2024

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model



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


def ARIMA_model(data, p = 0, d = 0, q = 0):
    # I've put this into its own function as it's done so often in the course
    # creating MA1 model. Note order = (AR, Integration, MA) or (p, d, q) as the industry standard
    model_ARIMA = ARIMA(data, order = (p, d, q))
    results_ARIMA = model_ARIMA.fit()
    results_ARIMA_resid = results_ARIMA.resid
    results_ARIMA_summary = results_ARIMA.summary()
    print(results_ARIMA_summary)
    print(results_ARIMA.params.round(3))
    #print(results_ARMA.pvalues)
    
    return model_ARIMA, results_ARIMA, results_ARIMA_resid


def ARIMAX_model(data, p = 0, d = 0, q = 0, exog = False):
    
    if isinstance(exog, bool):    
        model = ARIMA(data, order = (p, d, q))
    else:
        model = ARIMA(data, exog = exog, order = (p, d, q))
    
    results = model.fit()
    resid = results.resid
    summary = results.summary()
    print(summary)
    print(results.params.round(3))
    
    return model, results, resid


def SARIMAX_model(data, pdq: tuple, PDQs: tuple, exog = False):
    
    if isinstance(exog, bool):
        model = SARIMAX(data, order = pdq, seasonal_order = PDQs)
    else:
        model = SARIMAX(data, order = pdq, seasonal_order = PDQs, exog = exog)
        
    results = model.fit()
    resid = results.resid
    summary = results.summary()
    print(summary)
    print(results.params.round(3))
    
    return model, results, resid


def Clean_data(data):
    """Setting up data"""
    raw_csv_data = pd.read_csv(data)
    df_comp = raw_csv_data.copy()
    df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
    df_comp.set_index("date", inplace = True)
    df_comp = df_comp.asfreq("b")
    df_comp = df_comp.ffill()

    df_comp["Market value"] = df_comp.ftse
    
    size = int(len(df_comp)*0.8)
    df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

    df["returns"] = df["Market value"].pct_change(1).mul(100)
    df["sq_returns"] = df["returns"].mul(df["returns"])
    
    return df


def ACF_plot(data, title = "ACF plot", title_size = 18, lags = 40, ylim = 0.1, PACF = False):
    
    if PACF == False:
        sgt.plot_acf(data, zero = False, lags = lags)
    else:
        sgt.plot_pacf(data, zero = False, lags = lags)
    
    plt.title(title, size = title_size)
    plt.ylim(-1*ylim, ylim)
    plt.show()


def ARCH_model(data, update_freq: int = 1, mean = "AR", lags = 10, volatility_model = "GARCH", order_p = 1, order_q = 1, error_distribution = "normal"):
    
    # mean can equal "Constant", "Zero", or "AR"
    # AR says the mean is time-dependent, constant says the mean is constant, and zero says the mean is 0
    # Note: lags can be a list like [1, 2, 4, 5] to miss out lags which you don't think are significant
    # error_distribution is the probability distribution of the error terms could be "t", "ged" (for generalised error distribution), "normal" etc
    
    if mean == "AR":
        model = arch_model(data, mean = "AR", lags = lags, vol = volatility_model, p = order_p, q = order_q, dist = error_distribution)
    else:
        model = arch_model(data, mean = mean, vol = volatility_model, p = order_p, q = order_q, dist = error_distribution)
         
    results = model.fit(update_freq = update_freq)
    summary = results.summary()
    print(summary)
    
    return model, results
    

df = Clean_data("Index2018.csv")
print(df)

# plotting PACF for the sq_returns
ACF_plot(df["returns"][1:], title = "PACF of returns", ylim = 0.25, PACF = True)
ACF_plot(df["sq_returns"][1:], title = "PACF of sq_returns", ylim = 0.25, PACF = True)

# the first 6 lags of the sq_returns are very significant, which suggests that the data tends to see short-term trends of volatility

"""ARCH model introduction"""
GARCH1 = ARCH_model(df["returns"][1:])
# GARCH Model Results summary table shows the mean to be constant, which we'd expect for returns as they're stationary
# as we haven't passed in any parameters to this yet, this is actually using a GARCH model
# distribution is normal (this is the distribution of the residuals)
# the method describes how you find the coefficients. This is currently doing it via the 'maximum liklihood' method
# DF model shows the number of degrees of freedom this model has

# the Mean Model summary table shows the mean (mu) is 0.0466 with a high t-value

# the Volatility Model summary table is what we're mainly interested in
# Omega = constant value in the equation (is listed as a0 in my notebook)
# Alpha[1] = coefficient for the squared values (a1 in my notebook)
# Beta[1] = We'll come to this later

# There's also an iteration table. This is because multiple equations are being fit, so it takes multiple runs before the model converges
# Ie it fits a model with certain coefficients, checks how well it performs and then goes again
# It will generally stop once the Log Liklihood stops changing (LLF)

"""Simple ARCH model"""
# like I said above, the previous model wasn't actually an ARCH model, it was a GARCH model
# this is because we didn't set any parameters into the model. The above model assumes that the mean of the series is not serially correlated (ie the mean is time-invariant)
# because of this, it doesn't use past values or past residuals
ARCH1 = ARCH_model(df["returns"][1:], mean = "Constant", volatility_model = "ARCH", order_p = 1)

# only took 6 iterations, which suggests the model is light and doesn't take much computing
# Note: R^2 measurements are useful, but not for ARCH models (for some reason?)
# the log-liklihood value for this model is higher than the ARIMA ones we fitted previously, showing a simple ARCH model is better than the more
# complex ARIMA models

# the mean's p-value is on the order of e-02, so it's statistically significant
# In fact, all the three variables being used in the model are significant. As the log-liklihood is higher for this model compared to the ARIMAX
# ones we tried, this is the best one so far. HOWEVER: ARCH CAN ONLY BE USED TO PREDICT FUTURE VARIANCE, NOT FUTURE RETURNS

ARCH2 = ARCH_model(df["returns"][1:], mean = "Constant", volatility_model = "ARCH", order_p = 2)
# ARCH2 has all significant pvalues compared to ARCH1, and it has a higher log-liklihood value and lower AIC/BIC values (information criteria)
# ie ARCH2 is better than ARCH1

# if you keep doing this, you find that ARCH13 is the first one which breaks and a constant becomes insignificant (p=6.87e-02 or 0.0687)
ARCH13 = ARCH_model(df["returns"][1:], mean = "Constant", volatility_model = "ARCH", order_p = 13)

# the more terms of past squared-residuals we add, the less the model improves each time (makes sense right?), so the less important each one becomes
# (this is very similar to what happened with the MA model)
# with the MA model we then decided to use the MA model in combination with the AR model (ie the values)
# we can do something similar here...

"""GARCH: An ARMA equivalent for ARCH"""

# we can improve ARCH models by using past values as well
# but what actually would these past values be? Including past returns wouldn't make sense as these are already accounted for in the mean equation
# So instead, we'd use past conditional variances, which should help us explain current conditional variances

 
GARCH11 = ARCH_model(df["returns"][1:], mean = "Constant", order_p = 1, order_q = 1)
# a GARCH11 model gives a higher log-likilhood than the ARCH12 model - ie adding just one past conditional volatility value is better than adding 11 previous residual^2 values

"""Higher order GARCH models"""
# for returns, higher order GARCH models are not as effective as the GARCH11 model. Note how the p-value of Beta2 is 1 (ie it's very insignificant)
GARCH12 = ARCH_model(df["returns"][1:], mean = "Constant", order_p = 1, order_q = 2)
# this is the same for GARCH13
GARCH12 = ARCH_model(df["returns"][1:], mean = "Constant", order_p = 1, order_q = 3)


# if you do GARCH21, we see that one of the coefficients is not significant and this repeats
GARCH21 = GARCH12 = ARCH_model(df["returns"][1:], mean = "Constant", order_p = 2, order_q = 1)


