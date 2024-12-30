#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:30:06 2024

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
    
    return df


def ACF_plot(data, title = "ACF plot", title_size = 18, lags = 40, ylim = 0.1, PACF = False):
    
    if PACF == False:
        sgt.plot_acf(data, zero = False, lags = lags)
    else:
        sgt.plot_pacf(data, zero = False, lags = lags)
    
    plt.title(title, size = title_size)
    plt.ylim(-1*ylim, ylim)
    plt.show()
        
    

df = Clean_data("Index2018.csv")
ALL = False

if ALL == True:
    
    ARIMA111, ARIMA111_res, ARIMA111_resid = ARIMA_model(df["Market value"], 1, 1, 1)
    # note how the integration doesn't have any coefficients associated with it
    # this is because it's just transforming the data before applying a model
    
    ACF_plot(ARIMA111_resid[1:], title = "ARIMA111 Residuals ACF")
    # needs to remove the first element in the ACF as the ACF function computes the ACF from the previous element
    # so missing one data point will mean it's just 0 forever
    
    # Up to 4 lags is significant for now. From the lecture, it's ARIMA111, ARIMA112, ARIMA113, ARIMA211, ARIMA311
    # and ARIMA312 which have signifcant coefficients. As we know these are the only good options, we won't look at
    # the summary tables
    
    ARIMA112, ARIMA112_res, ARIMA112_resid = ARIMA_model(df["Market value"], 1, 1, 2)
    ARIMA113, ARIMA113_res, ARIMA113_resid = ARIMA_model(df["Market value"], 1, 1, 3)
    ARIMA211, ARIMA211_res, ARIMA211_resid = ARIMA_model(df["Market value"], 2, 1, 1)
    ARIMA311, ARIMA311_res, ARIMA311_resid = ARIMA_model(df["Market value"], 3, 1, 1)
    ARIMA312, ARIMA312_res, ARIMA312_resid = ARIMA_model(df["Market value"], 3, 1, 2)
    
    x = {"ARIMA112": ARIMA112_res, "ARIMA113": ARIMA113_res, "ARIMA211": ARIMA211_res, 
         "ARIMA311": ARIMA311_res, "ARIMA312": ARIMA312_res}
    
    for i, j in x.items():
        print(f"{i} - LLF: {round(j.llf, 3)}, AIC: {round(j.aic, 3)}")
        
    # ARIMA113 has the highest LLR and the lowest AIC so is the best model out of all of these
    # however, as ARIMA111 and ARIMA112 are nested in ARIMA113, we need to run the LLR test on these to make sure
    # that ARIMA113 is significantly better than the simpler models
    
    #LLR_test(ARIMA111, ARIMA113, DF = 2)
    #LLR_test(ARIMA112, ARIMA113, DF = 1)
    # ARIMA113 comes out on top of both of these
    
    ACF_plot(ARIMA113_resid[1:], title = "ACF of ARIMA113 residuals", ylim = 0.05)
    # The sixth lag appears to be significant, so there's likely a better model out there
    # So we must go through all the combinations from ARIMA111 - ARIMA616 to find out which ones could be options
    # The lecturer already did this and only two were possible options - ARIMA613 and ARIMA511
    
    ARIMA511, ARIMA511_res, ARIMA511_resid = ARIMA_model(df["Market value"], 5, 1, 1)
    # the lecturer said ARIMA613 is good but this is giving not good values at all
    ARIMA613, ARIMA613_res, ARIMA613_resid = ARIMA_model(df["Market value"], 6, 1, 3)
    
    x = {"ARIMA511": ARIMA511_res, "ARIMA613": ARIMA613_res}
    for i, j in x.items():
        print(f"{i} - LLF: {round(j.llf, 3)}, AIC: {round(j.aic, 3)}") 
    # again lecturer says the best model is the ARIMA613 but this is definitely not the best one but I'll carry on with
    # it, just so it matches the lecture
    
    LLR_test(ARIMA113, ARIMA613, DF = 5)
    # he ends up going with ARIMA511 anyway, as ARIMA613 doesn't do well enough in the LLR test to be worth it over ARIMA511
    ACF_plot(ARIMA511_resid[1:], title = "ARIMA511 Residual ACF", ylim = 0.05)
    # the ACF plot shows this is a good model, you could do more lags but we don't want to overfit the data

    """All the above code assumes that only 1 integration is needed to get the best model, but what if that's not true?
    Below will go over multiple integrations
    
    Q: How many integrations are needed? 
    A: You only want to integrate until the time series being analysed becomes stationary (as you can then
                                                                                           apply the ARMA mode)
    
    """
    
    # manually creating an integrated time series in the original dataframe to see if it's stationary
    # Note: you can check if you generated the data correctly by applying an ARMA11 model to it and checking
    # it matches the ARIMA111 model on the main data
    
    df["d_prices"] = df["Market value"].diff(1)
    
    # if df['d_prices'] is the correct dataset, these two models below should give ~identical outputs
    ARMA11 = ARIMA_model(df["d_prices"][1:], p = 1, d = 0, q = 1)
    ARIMA111 = ARIMA_model(df["Market value"][1:], 1, 1, 1)
    
    # now we run the Dicky-Fuller test to see if the data is stationary or not
    print(sts.adfuller(df.d_prices[1:]))
    # test statistic of -32.244, which is ~10x greater than the 1% value. p-value is also 0 so this shows the
    # data is very likely to be stationary. Therefore no other integrations are necessary
    
    # we're going to use the S&P500 and dax prices as the exogeneous variables to show how to do it
    
    """ARIMAX model"""
    
    ARIMAX111_Xspx = ARIMAX_model(df["Market value"], 1, 1, 1, exog = df["spx"])
    
    ARIMAX111_Xspx_dax = ARIMAX_model(df["Market value"], 1, 1, 1, exog = df[["spx", "dax"]])
    
else:
    pass


"""SARIMAX model"""

SARIMAX = SARIMAX_model(df["Market value"], pdq = (1, 0, 1), PDQs = (2, 0, 1, 5), exog = df["spx"])










