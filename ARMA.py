#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:53:38 2024

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
    
    print(f"\np-value of LLR test between {name1} & {name2} is: {p}")
    return p


def ARMA_model(data, AR = 0, MA = 0):
    # I've put this into its own function as it's done so often in the course
    # creating MA1 model. Note order = (AR, MA, Integration)
    model_ARMA = ARIMA(data, order = (AR, 0, MA))
    results_ARMA = model_ARMA.fit()
    results_ARMA_resid = results_ARMA.resid
    results_ARMA_summary = results_ARMA.summary()
    print(results_ARMA_summary)
    print(results_ARMA.params)
    #print(results_ARMA.pvalues)
    """
    if results == True:
        return model_ARMA, results_ARMA
    
    if params == True:
        return model_ARMA, results_ARMA.params, results_ARMA.pvalues
    
    elif resid == True:
        return model_ARMA, results_ARMA_resid
    
    else:
        return model_ARMA"""
    return model_ARMA, results_ARMA, results_ARMA_resid

    
def Clean_data(data):
    """Setting up data"""
    raw_csv_data = pd.read_csv(data)
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
    
    return df

df = Clean_data("Index2018.csv")


"""
Note: I've commented this all out so it's not running everytime


# creating the first ARMA model
ARMA11, x, y = ARMA_model(df["returns"], AR = 1, MA = 1)
# the AR1 term is positive, showing that prices tend to go in periods of increases followed by decreases etc
# the MA1 term is negative, indicating that we should be moving away from the previous value

# testing to see if the ARMA model is a better predictor than AR1 and MA1 models individually
AR1, x, y = ARMA_model(df.returns, AR = 1)
MA1, x, y = ARMA_model(df.returns, MA = 1)

'''
LLR_test(AR1, ARMA11) # note ARMA has two DF, while AR1 and MA1 have 1 each. The difference is 1 which is the default value
LLR_test(MA1, ARMA11)
'''
# p values are both 0, showing this is a better model

'''Higher lag ARMA models'''
# we use a slightly different approach when deciding which ARMA model to use
# this is because ARMA models SHOULD require fewer lags to predict a dataset well

# we start with an over-parameterised model, with several lags for both the AR & MA components
# we then start decreasing the lags 

# since the PACF and ACF graphs gave us a good idea of what models to use when using only AR/MA models
# so it'd be a good idea to start from these graphs

sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title("ACF of Returns", size = 20)
plt.ylim(-0.1, 0.1)
plt.show()

sgt.plot_pacf(df.returns[1:], zero = False, lags = 40)
plt.title("PACF of Returns", size = 20)
plt.ylim(-0.1, 0.1)
plt.show()

# graphs indicate that we'd expect the ARMA model to have no more than 6 AR components and 8 MA components
# (as these are the significant values from the PACF/ACF plots)

# starting with the over complicating model
#test_model = ARMA_model(df.returns[1:], AR = 6, MA = 8)
# if you run this ^, you find that of the 14 component coefficients, 7 are non-significant showing the model is too complicated
# (it takes a while to run though which is why it's commented out)


def Optimial_ARMA_Components(data):
    
    # sets initial components at one, it will work up until too many results become insignificant
    AR = 1
    MA = 1
    
    model, params, pvalues = ARMA_model(data, AR = AR, MA = MA, params = True)
    
    AR_components = [i for i in pvalues.index if abs(pvalues[i]) < 0.05]
    
    print(AR_components)


# we'll start with a 3,3 model
#ARMA33 = ARMA_model(df.returns[1:], 3, 3)
#LLR_test(ARMA11, ARMA33, DF = 4) # DF = 4 as ARMA11 has 2 DoF while ARMA33 has 6

# ARMA33 is a much better model than ARMA11, but AR has an insignificiant components
# this hints that the most optimal model is between ARMA11 and ARMA33

ARMA32, ARMA32_res, ARMA32_resid = ARMA_model(df.returns[1:], 3, 2)
# this looks good as the p-values are all fine and for both the AR and MA components, their magnitude
# decreases with successive lags, indicating they're less important (which fits our initital assumption)

# Note: we don't run the LLR test between ARMA33 and ARMA32 because ARMA33 had insignificant coefficients
# whereas ARMA32 didn't, so we know ARMA32 is a better model

#ARMA23 = ARMA_model(df.returns[1:], 2, 3)

# CANT USE THE LLR TEST BETWEEN ARMA23 AND ARMA32 AS THEY HAVE THE SAME DF
#LLR_test(ARMA23, ARMA33, DF = 1)
# p-value of 0.042 so we should opt for ARMA33 out of these two models

#ARMA31 = ARMA_model(df.returns[1:], 3, 1)
#LLR_test(ARMA31, ARMA32, DF = 1)
# more complicated model is better (p-value = 0.01)

#ARMA22 = ARMA_model(df.returns[1:], 2, 2)
# values are much worse in ARMA22

#ARMA13, ARMA13_res = ARMA_model(df.returns[1:], 1, 3, results = True)
# Note: we can't use the LLR test here because ARMA13 and ARMA 33 are nested, so we do it manually instead

print(ARMA32_res.llf, ARMA32_res.aic)
#print(ARMA13_res.llf, ARMA13_res.aic)

# best model seems to be the ARMA32 - has all non-significant coefficients and outperforms all other, simpler, models

# doing the ARMA12 and ARMA21 give non-significant coefficients so I won't do those here

'''Analysing the residuals'''
df["ARMA32_resid"] = ARMA32_resid

print(df)
df.ARMA32_resid.plot()
plt.title("ARMA32 Residuals", size = 20)
plt.show()

sgt.plot_acf(df.ARMA32_resid[1:], zero = False, lags = 40)
plt.title("ACF of ARMA32 Residuals", size = 18)
plt.ylim(-0.05, 0.05)

# We find that the 5th lag is significant so we should check ARMA55, ARMA5Q, and ARMAP5 to see if including
# them could improve the model. (ie ARMA51, ARMA52... ARMA55 and ARMA15, ARMA25... ARMA55)
# doing this only gives ARMA51 and ARMA15 being potential better models
# they have the same DF so an LLR test can't be used

ARMA51, ARMA51_res, ARMA51_resid = ARMA_model(df.returns[1:], 5, 1)
#ARMA15 = ARMA_model(df.returns[1:], 1, 5)

# ARMA51 is the better model of these two
# now which is best between the ARMA32 and ARMA51?

LLR_test(ARMA32, ARMA51)
# ARMA51 is a better model than ARMA32 as it has a pvalue of 0.00033
df["ARMA51_resid"] = ARMA51_resid

df.ARMA51_resid.plot()
plt.title("ARMA51 Residuals", size = 20)
plt.show() 

sgt.plot_acf(df.ARMA51_resid[1:], zero = False, lags = 40)
plt.title("ACF of ARMA51 Residuals", size = 18)
plt.ylim(-0.05, 0.05)
"""

# some residuals are significant but again they're over 10 lags ago so shouldn't be important under 
# our assumptions

"""ARMA models with non-stationary data (ie prices)"""

ARMA11, ARMA11_res, ARMA11_resid = ARMA_model(df["Market value"], 1, 1)

df["ARMA11_resid"] = ARMA11_resid

# examining the residuals before choosing the over-parameterised model
sgt.plot_acf(df.ARMA11_resid[1:], zero = False, lags = 40)
plt.title("ACF of ARMA11 Residuals", size = 18)
plt.ylim(-0.1, 0.1)

# error terms show that we should try ARMA66 

#ARMA66 = ARMA_model(df["Market value"], 6, 6)
# all the coefficients in ARMA66 are non-significant, so now we'd have to go through all the models 
# (ARMA61-ARMA66 and ARMA16-ARMA66) to find ones with all significant coefficients and then find the best
# of those. Doing this, you get ARMA56 and ARMA61 are the only two candidates for a good model

#ARMA56, ARMA56_res, ARMA56_resid = ARMA_model(df["Market value"], 5, 6)
ARMA61, ARMA61_res, ARMA61_resid = ARMA_model(df["Market value"], 6, 1)

#LLR_test(ARMA61, ARMA56, DF = 4) # ARMA61 is better from the LLR test

df["ARMA61_resid"] = ARMA61_resid

sgt.plot_acf(df.ARMA61_resid[1:], zero = False, lags = 40)
plt.title("ACF of ARMA61 Residuals", size = 18)
plt.ylim(-0.05, 0.05)

# if you compare the LLR values and AIC values for the best ARMA models for prices and returns, you see that
# ARMA models can be used to model non-stationary data, but they aren't anywhere near as good compared to their
# ability to predict stationary data



