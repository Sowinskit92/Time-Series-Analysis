#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:26:33 2025

@author: timsowinski
"""

import statsmodels
import pandas as pd
import os
import scipy
import statsmodels.api as sm
import seaborn as sns
import sklearn
import yfinance
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
from pmdarima.arima import auto_arima
from datetime import datetime
from statsmodels.tsa.api import VAR




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
    

def Yfinance_Data_Gather(old = False):
    
    file_name = "yfinance_data.csv"
    if old == True:
        if file_name not in os.listdir():
            raw_data = yfinance.download(tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", end = "2019-09-01",
                                         interval = "1d", group_by = "ticker", auto_adjust = True, threads = True)
            df = raw_data.copy()
            
            cols = {"^GSPC": "spx", "^GDAXI": "dax", "^FTSE": "ftse", "^N225": "nikkei"}
            
            print(raw_data)
            # gets only the closing values from each of the indicies
            for i, j in cols.items():
                print(i, j)
                df[j] = df[i].Close[:]
                del df[i]
            
            df = df.asfreq("b")
            df = df.ffill()
            
            # creates returns columns
            for i in cols.values():
                df[f"ret_{i}"] = df[i].pct_change(1).mul(100)
            
            print(df)
            
            df.to_csv(file_name)
        else:
            df = pd.read_csv(file_name)
        
        return df
    else:
        
        raw_data = yfinance.download(tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", end = "2019-09-01",
                                     interval = "1d", group_by = "ticker", auto_adjust = True, threads = True)
        
        df_comp = raw_data.copy()
        
        x = {"^GSPC": "spx", "^GDAXI": "dax", "^FTSE": "ftse", "^N225": "nikkei"}
        
        for i, j in x.items():
            df_comp[j] = df_comp[i].Close[:]
        
        
        df_comp = df_comp.iloc[1:] # not sure why they've done this
        
        for i in x.keys():
            del df_comp[i]
        
        df_comp = df_comp.asfreq("b") # business days
        df_comp = df_comp.ffill()
        
        # =============================================================================
        # Creating returns     
        # =============================================================================
        
        for i in x.values():
            df_comp[f"ret_{i}"] = df_comp[i].pct_change(1).mul(100)
            df_comp[f"norm_ret_{i}"] = df_comp[f"ret_{i}"].div(df_comp[f"ret_{i}"][1])*100
        
        print(df_comp)
        
        # =============================================================================
        # Splitting the data    
        # =============================================================================
        
        size = int(len(df_comp)*0.8)
        df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
        
        return df, df_test, df_comp
        
    


"""
In here, we'll be forecasting using a simple AR model
"""

df, df_test, df_comp = Yfinance_Data_Gather()

"""
#df_comp = df_comp.asfreq("b")
size = int(len(df_comp.index)*0.8)

df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]"""

#print(df)

start_date = "2014-07-15"
end_date = "2015-01-01"

'''
# creating model, first with just the default values (which aren't usually optimal for a problem...)
AR1, results_AR1, resid_AR1 = ARIMA_model(df.ftse, p = 1)

"""The forecast"""
# we set the starting date of the forecast to the first date we don't have values for
#print(df.tail())


"""df_pred = results_AR1.predict(start = start_date, end = end_date)

df_pred[start_date:end_date].plot(color = "red")
df_test.ftse[start_date:end_date].plot(color = "blue")
plt.title("Price Predictions vs Actual - AR1")
plt.show()
"""

"""Due to the simple nature of the model, this is not a brill prediction
+ it's also non-stationary so the AR model will be even worse"""

# =============================================================================
# Now trying the returns
# =============================================================================

Just commenting this out so it doesn't keep running

# creating model, first with just the default values (which aren't usually optimal for a problem...)
AR1_ret, results_AR1_ret, resid_AR1_ret = ARIMA_model(df.ret_ftse[1:], p = 5)
df_pred_ret_AR1 = results_AR1_ret.predict(start = start_date, end = end_date)

"""
df_pred_ret[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Return Predictions vs Actual - AR1")
plt.show()
"""
# our model shows essentially no predictions. Checking the summary table shows that the coefficients
# for the model aren't even significant!
# Also, if you up the AR order, you see slight changes in the returns at the beginning but they go in 
# the exact opposite direction to the actual values (which makes sense!)

# =============================================================================
# Now trying the MA model
# =============================================================================

MA1_ret, results_MA1_ret, resid_MA1_ret = ARIMA_model(df.ret_ftse[1:], p = 0, q = 3)
df_pred_ret_MA1 = results_MA1_ret.predict(start = start_date, end = end_date)

"""
df_pred_ret_MA1[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Return Predictions vs Actual - MA1")
plt.show()
"""

"""Again, the model is a bit rubbish"""
print(df_pred_ret_MA1.head(5))
"""You can see that after the first few dates, the prediction value doesn't change
If you up the lags, you notice that this happens until the q-th day and then goes to a constant.
This suggests there are no longer error terms in the data? But how can there be no error terms?
"""
# =============================================================================
# Now trying the ARMA model
# =============================================================================
ARMA1_ret, results_ARMA1_ret, resid_ARMA1_ret = ARIMA_model(df.ret_ftse[1:], p = 2, q = 1)
df_pred_ret_ARMA1 = results_ARMA1_ret.predict(start = start_date, end = end_date)
"""
df_pred_ret_ARMA1[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.show()
"""
#print(df_pred_ret_ARMA1.head(5))
#print(df_pred_ret_ARMA1.tail(5))

"""The ARMA provides more reasonable predictions as they don't just die off immediately, but 
they still aren't great. Changing the orders doesn't really affect anything. 
"""
# =============================================================================
# Now trying the ARMAX model
# =============================================================================
ARMAX1_ret, results_ARMAX1_ret, resid_ARMAX1_ret = ARIMAX_model(df.ret_ftse[1:], p = 1, q = 1, 
                                                                exog = df[["ret_spx", "ret_dax", "ret_nikkei"]][1:])

df_pred_ret_ARMAX1 = results_ARMAX1_ret.predict(start = start_date, end = end_date, exog = df_test[["ret_spx", "ret_dax", "ret_nikkei"]][start_date:end_date])
"""
df_pred_ret_ARMAX1[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.show()
"""



"""ARMAX is a much better predicter than the other models but we can't use it for actual forecasting
as we don't know the exogeneous variables into the future"""


ARIMA11, results_ARIMA11, resid_ARIMA11 = ARIMA_model(df.ftse[1:], p = 2, d = 1, q = 3)

df_pred_ARIMA11 = results_ARIMA11.predict(start = start_date, end = end_date)
"""
df_pred_ARIMA11[start_date:end_date].plot(color = "red")

df_test["ftse_diff"] = df_test["ftse"].diff(1)
# you need to create the 'integrated' version of the dataset to graph it against
df_test.ftse_diff[start_date:end_date].plot(color = "blue")
"""

"""ARIMA doesn't work too well either"""

ARIMAX11, results_ARIMAX11, resid_ARIMAX11 = ARIMAX_model(df.ftse[1:], p = 1, d = 1, q = 1, 
                                                          exog = df[["spx", "dax", "nikkei"]][1:])

df_pred_ARIMAX11 = results_ARIMAX11.predict(start = start_date, end = end_date, exog = df_test[["spx", "dax", "nikkei"]][start_date:end_date])


df_pred_ARIMAX11[start_date:end_date].plot(color = "red")
df_test.ftse[start_date:end_date].plot(color = "blue")



# we set s = 5 for 5 business days. It makes sense that the non-seasonal orders are
# less than the seasonal order (otherwise the seasonal one will be counted twice)
SARMA, results_SARMA, resid_SARMA = SARIMAX_model(df.ret_ftse[1:], pdq = (3, 0, 4),
                                                           PDQs = (3, 0, 4, 5))

df_pred_SARMA = results_SARMA.predict(start_date = start_date, end_date = end_date)

df_pred_SARMA[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")

# =============================================================================
# SARIMAX model
# =============================================================================

SARIMAX, results_SARIMAX, resid_SARIMA = SARIMAX_model(df.ret_ftse[1:], pdq = (3, 0, 4),
                                                       PDQs = (3, 0, 2, 5), exog = df[["ret_spx", "ret_dax", "ret_nikkei"]][1:])

df_pred_SARIMAX = results_SARIMAX.predict(start = start_date, end = end_date,
                                          exog = df_test[["ret_spx", "ret_dax", "ret_nikkei"]][start_date:end_date])

df_pred_SARIMAX[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")


# =============================================================================
# Auto ARIMA
# =============================================================================
# the auto-ARIMA should work better than it does, not really sure why it isn't...
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[["ret_spx", "ret_dax", "ret_nikkei"]][1:], 
                        m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5)

# note: the auto_arima model doesn't return a dataframe, it returns a numpy array
# it also doesn't take start/end dates - it takes n_periods instead. An index also needs to be supplied

df_auto_pred = pd.DataFrame(model_auto.predict(n_periods = len(df_test[start_date:end_date].index),
                                               exogenous = df_test[["ret_spx", "ret_dax", "ret_nikkei"]][start_date:end_date]),
                            index = df_test[start_date:end_date].index)

"""NOTE: IT'S SPELT EXOGENOUS, IF IT's SPELT WRONG LIKE I KEEP DOING THE MODEL WILL JUST IGNORE IT"""
df_auto_pred.plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")



# =============================================================================
# Forecasting volatility
# =============================================================================

# now using the full model as this doesn't forecast values 
GARCH = arch_model(df_comp.ret_ftse[1:], vol = "GARCH", p = 1, q = 1, mean = "constant", dist = "Normal")

# setting the last observation to start_date, so it uses the training set instead of the whole thing
res_GARCH = GARCH.fit(last_obs = start_date, update_freq = 10)

# horizon is how many observations we want the model to predict for each date
# setting it to 1 means we'll get the forecasted value for the next date, if it was 2 it'd give values for the next date and the one after that

# align determines whether the model matches the values with the dates the prediction is made on ("origin")
# or the date the forecast is representing ("target")
pred_GARCH = res_GARCH.forecast(horizon = 1, align = "target")
pred_GARCH.residual_variance[start_date:].plot(color = "red", zorder = 2)

# we can plot the abs returns of ftse to see how well the testing model does (abs as variance is always positive)
df_test.ret_ftse[start_date:].abs().plot(color = "blue", zorder = 1)

# note: zorder determines which line is graphed on the top
plt.title("Volatility predictions")


# now moving onto forecasting the volatility rather than predicting it
# we set horizon to a large number (100 in this example) and then we graph 
pred_GARCH = res_GARCH.forecast(horizon = 100, align = "target")
GARCH_forecast = pred_GARCH.residual_variance[-1:]
print(GARCH_forecast)

# all you'd need to do to graph this volatility forecast is melt the data and reset the index to
# be days, then you could plot it as normal

'''
# =============================================================================
# Multi-variate Regression
# =============================================================================

df_ret = df[["ret_spx", "ret_dax", "ret_ftse", "ret_nikkei"]][1:]

# creating the model
model_VAR_ret = VAR(df_ret)
# select_order - the more time series we add, the higher this should be
# from my understanding, it's the max number of total components of the model (eg p, q but each time series would have some of these)
model_VAR_ret.select_order(20)
results_VAR_ret = model_VAR_ret.fit(ic = "aic")
print(results_VAR_ret.summary())


lag_order = results_VAR_ret.k_ar # highest lag order

print(df_ret)
# df.values turns a dataframe into a numpy array
# we're using the last n = lag_order (which is 5 in this example) values to make our prediction
print(df_ret.values[-lag_order:])
VAR_pred_ret = results_VAR_ret.forecast(df_ret.values[-lag_order:], len(df_test[start_date:end_date]))
print(df_test.columns.tolist())

df_pred_ret = pd.DataFrame(data = VAR_pred_ret, index = df_test[start_date:end_date].index, 
                           columns = ["ret_spx", "ret_ftse", "ret_dax", "ret_nikkei"])

print(df_pred_ret)

df_pred_ret.ret_nikkei[start_date:end_date].plot(color = "red")
df_test.ret_nikkei[start_date:end_date].plot(color = "blue")

"""As you can see, this doesn't work well either, mainly because it's only using past values of the 
other time series, not the values for the forecasting day (which is why the MAX models worked so well)
"""

# we can use the plot_forecast() method which will give us the value + 2 standard deviations interval
# the forecasted bit are the lines on the right side. These can be used to give a good idea of the 
# stability in the market

results_VAR_ret.plot_forecast(1000)














