#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:48:14 2024

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
# import arch

"""Things to research further:
    1. How the Dicky-Fuller test works
    2. T-statistic
    3. What exactly is a lagged version of a dataset? (Done)
    4. How is the PACF calculated?
    5. What are the different ways of calculating a PACF?
    
"""


"""Now we've looked at ACF and PACF, we're going to take a look at choosing the best model for the data
you're working with


When creating a model, you 
1. Start with a simple one and then build up
2. When adding bits to the model, the new coefficients for the more complex model need to be significantly
different from 0. If not, they don't help us estimate future values, so should be omitted
3. You want your model to be Parsimonious (as simple as possible), unless a more complex one provides
much better predictions (to determine whether one is statistically better, there is something known as 
a Log-Liklihood Ratio (LLR) test, but this can only be applied to models with different degrees of freedom)

If you're trying to find out which model is best between two which have the same number of degrees of freedom, 
you need to look at the information criteria (ie how much data is needed for an accurate result). You can
further test using AIC and BIC - the model with the lower AIC/BIC coefficients uses less data for an
accurate prediction and therefore should be used

The other thing to look at is, if you have an accurate model, there should be no trend you haven't
accounted for, so the residuals should resemble white noise.

"""

"""Intro to AutoRegressive (AR) model"

This is a linear model where current period values are a sum of past outcomes multiplied by a numeric 
factor

Eg: x(t) = C0 + (Y1 * x(t-1)) + E(t)
where C0 = some constant, Y1 = numeric coefficient, E(t) = residual (error) at time t (diff. between our
prediction of x(t) and the actual value). These should be unpredictable as any other pattern should be 
accounted for in the model already
Note: The Y coefficients should always be between -1 and 1

The layout of this model will change with how many lags you need to use. Ie if you wanted to use 2 lags,
the equation would be: x(t) = C + Y1*x(t-1) + Y2*x(t-2) + E(t) 

The more lags we have, the more complicated and therefore, the more likely some will not be significant
"""

df = pd.read_csv(os.getcwd() + "/Index2018.csv")
df["date"] = pd.to_datetime(df["date"], format = "%d/%m/%Y")
df.set_index("date", inplace = True)
df = df.asfreq("b")

df["spx"] = df["spx"].ffill() # front fill
df["ftse"] = df["ftse"].bfill() # back fill
df["dax"] = df["dax"].fillna(value = df["dax"].mean()) # single value

size = int(len(df.index)*0.8) # sets size for the training/testing dataframes 

#df_train = df
df_train = df.iloc[:size]
df_test = df.iloc[size:]

"""From now on we'll only be looking at the FTSE100"""

df_train["Market value"] = df_train["ftse"]

print(df_train)

sgt.plot_acf(df_train["Market value"], zero = False, lags = 40)
plt.title("FTSE ACF", size = 24)
plt.show()

"""Sometimes by having too many lags, it can cause the model to fit the data too well and it won't work
well when moved on to different data sets"""

"""As we want an efficient model, we only want to include past lags which have a direct, significant
 effect on the model"""
 
sgt.plot_pacf(df_train["Market value"], zero = False, lags = 40, alpha = 0.05, method = ("ols"))
plt.title("FTSE PACF", size = 24)
plt.ylim(-0.1, 0.1)
plt.show()

"""Looking at the PACF, after 25 lags, the coefficients aren't significant, so we might as well remove
 them and only consider 25
 
An interesting point is that after 22 lags, the coefficients become mostly negative and this is
potentially likely due to the fact that each month there is ~22 business days so some of the values
from last month have a negative impact on the current value (noting the frequency of these data is 
in business days). But it's important not to overshadow their impacts, as they're very much 
outweighed by other coefficients

The first coefficient is greatly significant so it must be included in our model. Now we'll put this
into a simple AR1 model (as it uses 1 cofficient)

"""

"""This is what the video told me to do, but it's depreciated
from statsmodels.tsa.arima_model import ARMA

# for the order argument, the (1,0) the 1 represents the number of past values to include in the model
# and the 0 represents the fact we aren't going to include the residuals into consideration (why will
# be covered in the next section of the course)

model_ar = ARMA(df_train["Market value"], order = (1, 0)) # creates model
results_ar = model_ar.fit() # fits model to data
AR1_summary = results_ar.summary()
print(AR1_summary)
"""

from statsmodels.tsa.arima.model import ARIMA

model_AR = ARIMA(df_train["Market value"], order = (1,0, 0)) # creates model
results_AR = model_AR.fit() # fits model
AR1_summary = results_AR.summary()

print(AR1_summary)

"""In AR1_summary, there's lots of info. The bits we're going to focus on for now is the table 
~half way down which tells us the value of the constant and coefficient (const = 5089.69 and 
 ar.L1 = 0.9984) (ar.L1 = Autoregressive value from one lag ago)
"""

"""Got to 14:24 on https://www.youtube.com/watch?v=hprO9_VtKso&list=PLtIY5kwXKny91_IbkqcIXuv6t1prQwFhO&index=7
"""
 

