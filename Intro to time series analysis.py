#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:13:06 2024

@author: timsowinski

This is all from https://www.youtube.com/playlist?list=PLtIY5kwXKny91_IbkqcIXuv6t1prQwFhO

"""

import statsmodels
import pandas as pd
import os
import scipy.stats
import pylab
import matplotlib.pyplot as plt
import numpy as np
import sys
# import arch

"""Things to research further:
    1. How the Dicky-Fuller test works
    2. T-statistic
    3. What exactly is a lagged version of a dataset? (Done)
    4. How is the PACF calculated?
    5. What are the different ways of calculating a PACF?
    
"""


"""Conventions"""

# capital letters are used for time series variables, eg stock value as a function of time
# T = the time period covered by a time series
# t = a single element of a T

"""Important notes"""

"""
1. Intervals in time series need to be identical
2. Missing data needs to be dealt with properly, as values tend to affect each other from one period
to the next
3. Changing the frequency of data will likely require modelling new values which aren't in the original
data source (eg if you had monthly data, but wanted to know something in days)
4. When it comes to ML algorithms, you pick a cut off time to train your model, after this you test the model

"""

# loads data from session 1
# is just market values of the S&P500, FTSE100, DAX30, and Nikkei as a time series
df = pd.read_csv(os.getcwd() + "/Index2018.csv")

# need to check for missing elements in time series analysis
num_missing_elements = df.isna().sum()
print(num_missing_elements)

fig1, ax1 = plt.subplots()
#fig1.set_title("Stock market prices")
# df['spx'].plot(title = "S&P500 prices", figsize = (20, 5))
for i in df.columns.tolist()[1:]:
    ax1.plot(df[i])
plt.title("Stock prices")
"""The Quantile-Quantile (QQ) plot

This is used to show which values occur most frequently. By default it will compare to a normal
distribution

QQ plot takes all the values a variable can take and arranges them in ascending order

In the case coded here, the y-axis shows the prices. The x-axis shows how many standard deviations
away from the mean these values are.

The red line shows the trend the data points should follow if they were normally distributed.

In this dataset, the data does not follow the normal distribution very well, because of the many values
at ~500, so we can't just use normal distribution maths in creating a forecast
"""

fig2, ax2 = plt.subplots()

scipy.stats.probplot(df["spx"], plot = pylab)
pylab.show()


df["date"] = pd.to_datetime(df["date"], format = "%d/%m/%Y")
df.set_index("date", inplace = True)

print(df)

"""Setting the frequency of data

asfreq() takes letter arguments. d = daily, h = hourly, w = weekly, m = monthly, a = annual etc"""

df = df.asfreq("d")
print(df)

# checking for missing data again you can see there's lots now, because the index has been updated for 
# each day even if there isn't any data
num_missing_elements = df.isna().sum()
print(num_missing_elements)

df = df.asfreq("b") # sets frequency to business days (ie removes weekends/bank holidays)
print(df)

num_missing_elements = df.isna().sum()
print(num_missing_elements) # much fewer missing values now! But still some, so need to deal with them

"""Dealing with missing values"""

"""fillna can be used in a few different ways. fillna begins at the start of the timeseries and goes down

1. Front filling - assigns an na value the previous non-na value
2. Back filling - assigns an na value the next non-na value
3. Assigning the same value to all na values in the time series
"""

# df["spx"] = df["spx"].fillna(method = "ffill") # note this is depreciated now
df["spx"] = df["spx"].ffill() # front fill
df["ftse"] = df["ftse"].bfill() # back fill
df["dax"] = df["dax"].fillna(value = df["dax"].mean()) # single value

"""From now on we'll only be looking at the S&P500"""

df["Market value"] = df["spx"]

df = df[["Market value"]]
print(df)


"""Splitting data to create and test models"""

"""It's important to note that training sets can be too long otherwise your model will fit the data too
well and probably perform poorly, but it also can't be too short as it won't be accurate enough. 

I'll assume a 80-20 split between training and testing"""

size = int(len(df.index)*0.8) # sets size for the training/testing dataframes 

df_train = df.iloc[:size] 
df_test = df.iloc[size:]

print(df_train.tail(1)) # check there's no overlapping value, which there isn't, so that's good
print(df_test.head(1))

"""White noise"""

"""
White noise is data which doesn't follow a pattern. The assumption of timeseries data is that past
patterns/drivers will happen again in future, so you want to remove as much white noise from your data
as possible

For data to be considered as white noise, it needs to satisfy the following three conditions:
1. Having a constant mean
2. Having a constant variance (std)**2
3. Having no auto correlation in any period, ie no clear relationship between past/present values in the series

Auto-correlation measures how correlated a series is with past versions of itself

Basically it's erratic and can't be modelled

White noise often has large jumps between consecutive values in the time series

"""

white_noise = np.random.normal() # generates white noise data from a normal distribution

# to be comparable to the data though, we should get the mean + standard deviation of the white noise
# data to be the same as the S&P500 data

# sets position of the random data's mean and dispersion of the white noise to those of the S&P500 data
white_noise = np.random.normal(loc = df_train["Market value"].mean(), scale = df_train["Market value"].std(),
                                 size = len(df_train.index))

df_train["White noise"] = white_noise
df_train = df_train[["White noise", "Market value"]]
print(df_train.describe())

fig3, ax3 = plt.subplots()

for i in df_train.columns.tolist():
    df_train[i].plot()
plt.title("White noise")

# setting the y-axes limits to match on both the price and white noise graphs, would be used if they were plotted on different graphs

# plt.ylim(0, 2300)
"""Random walk

A type of time series where values tend to persist over time and the differences between values at
consecutive periods are white noise

eg. P(t) = P(t-1) + E, where P(t) is a price at time t, and E is a residual of white noise

In the example two lines above, this would suggest that the best predictor of today's prices will be 
yesterday's prices

"""

rw = pd.read_csv("Randwalk.csv")
print(rw)
rw["date"] = pd.to_datetime(rw["date"], format = "%d/%m/%Y")
rw.set_index("date", inplace = True)
rw.asfreq("b")
print(rw)

df_train["rw"] = rw["price"]

print(df_train)

fig4, ax4 = plt.subplots()

for i in df_train.columns.tolist()[1:]:
    df_train[i].plot()

"""Random walk is much more similar to S&P500 data rather than the white noise"""


"""Market efficiency

Is the measure of how difficult it is to forecast accurate future values

If a time series resembles a random walk, the timeseries can't be predicted with great accuracy

If it can be predicted with accuracy, this gives the potential for arbitrage 

Arbitrage - buying/selling commodities while making a safe profit as prices change

"""


"""Stationarity

Time series stationarity implies that taking consecutive series of data with the same size should
have identical covariances, regardless of the starting point. This is also known as 'weak-form' stationarity
or 'covariance' stationarity

A time series can be classifed as covariance stationarity if it satisfies three conditions:

1. It has a constant mean
2. It has a constant variance (std)**2
3. It has constant covariance over different periods of the same size
(ie Cov(x_n, x_n+k) = Cov(x_m, x_m+k))

Note: Cov(x_n, x_n+k) = Corr(x_n, x_n+k)*std1*std2

White noise is an example of this

To forecast accurately, you need to know if data is stationarity or non-stationarity

"""


"""Dickey-Fuller (DF) test

This is a method to determine if a time series is stationarity or not. It uses a null (H0) and 
alternative hypothesis (H1)

H0 assumes non-stationarity - assumes that the 1-lag auto-correlation coefficient is < 1
H1 looks like it assumes the 1-lag autocorrelation coefficient = 1

We compute the test statistic and compare it to a critical value from a Dicky-Fuller (DF) table
If test statistic < critical value -> data is stationarity

How to implement this in python is below

"""
import statsmodels.tsa.stattools as sts
DF_table = sts.adfuller(df_train["Market value"])
print()
print(DF_table)

"""The output of the Dicky-Fuller (DF) test are:
    (-1.7369847452352418, 0.41216456967706294, 18, 5002, {'1%': -3.431658008603046, '5%': -2.862117998412982, '10%': -2.567077669247375}, 39904.880607487445)

The first element (-1.7369...) is the critical value
The 1%, 5%, and 10% are the critical values at different intervals. The critical value is greater than
all of these, so it is evidence that the data is non-stationarity and therefore can be forecast

The second element (0.41216) is the p-value associated with the T-statistic. This is saying there's
~40% chance of not rejecting the null (ie that the data is stationarity), so we can't confirm it is
or isn't

The third element (18) is the number of lags used in computing the T-statistic, ie there's auto-correlation
going back 18 periods

The fourth argument (5002) is the number of observations used in the analysis

The fifth element is the maximised information criteria provided there's some auto-correlation. The
lower this number the easier it should be to forecast the dataset

For the white noise dataset:
"""

DF_table_wn = sts.adfuller(df_train["White noise"])
print(DF_table_wn)

"""In this case, the output is
(-70.66253346239897, 0.0, 0, 5020, {'1%': -3.431653316130827, '5%': -2.8621159253018247, '10%': -2.5670765656497516}, 70753.36883767006)

Some points:
    1. The critical value is much higher than the test statistics
    2. There's no lags used, as there's no auto-correlation in white noise
    3. The p-value is ~0, ie a ~0% chance it's non-stationarity

For the random walk dataset:
"""
DF_table_rw = sts.adfuller(df_train["rw"])
print(DF_table_rw)

"""In this case, the results are
(-1.3286073927689717, 0.6159849181617385, 24, 4996, {'1%': -3.4316595802782865, '5%': -2.8621186927706463, '10%': -2.567078038881065}, 46299.333497595144)

Some points:
    1. From the p-value, there's over a 60% chance this data comes from a non-stationarity process
    
However, this is exactly what you'd expect. Due to chance, random walk data will have periods of ups/downs
followed by periods of all ups or all downs. Therefore the covariances of two intervals of the same size
will VERY RARELY be equal, so it is likely these will be picked up as a non-stationarity process
"""

"""Seasonality
This refers to certain things having cyclical effects on a dataset (ie temperature etc)
One way to handle this is to decompose the sequence, where you split the timeseries into three effects:
1. The trend - the trend consistent throughout the data
2. The seasonal - all cyclical effects due to seasonality
3. The residual - the error of prediction between our actual data and the fitted model

The simplist type of decomposition is known as Naive Decomposition, we expect a linear relationship
between the three parts and the timeseries. Within this there are two types of Naive Decomposition:
    1. Additive - assumes observed value = trend + seasonal + residual
    2. Multiplicative - assumes observed value = trend * seasonal * residual

"""
# this package will split a timeseries into those three parts
from statsmodels.tsa.seasonal import seasonal_decompose

s_dec_add = seasonal_decompose(df_train["Market value"], model = "additive")
s_dec_add.plot()

"""The trend closely resembles the observed data. Remember that current period prices are the best
predictor of the next period's prices. If we observed seasonal patterns, we'll have other prices as
better predictors, eg if prices are consistently higher at the beginning of the month, you'd have better
luck using values from 30 days ago

The seasonal aspect is oscillating between 0.1 and -0.2 over and over, which gives the rectangle. This
means there's no concrete seasonal aspect to this data

The residuals are the errors in the Naive decomposition model, ie the difference between the model
value and the actual value. Note the big jumps in error around the 2008 financial crisis"""

s_dec_mul = seasonal_decompose(df_train["Market value"], model = "multiplicative")
s_dec_mul.plot()

"""The multiplicative one also has no seasonal pattern, so using these basic models, it doesn't
appear as though there's any seasonal relationship"""


"""Autocorrelation
Correlation measures the similarity between two variables, but with timeseries you only have the 
one variable/series. To figure out the correlation of past/future points, you need a new concept:
Autocorrelation

Autocorrelation represents the relation between the sequence and itself. More precisely, it measures
the level of resemblence between a sequence from several periods ago and the actual data. This is known
as lag

Autocorrelation essentially tells you how much of yesterday's values resembles today's values

To calculate the autocorrelation, we can use the AutoCorrelation Function (ACF). This calculates the 
autocorrelation value for however many lags you're interested in simultaneously

"""
import statsmodels.graphics.tsaplots as sgt
# note the lag's default argument is the length of the timeseries but this can take a long time if the
# timeseries has thousands of data points
# apparently, it's common practise to set the lags equal to 40
# By 40 lags, we mean 'the last 40 values from the current one'
# zero is False because we don't need to include the current period values in the graph
# we don't need this as the autocorrelation value of a value and itself will always be 1
sgt.plot_acf(df_train["Market value"], lags = 40, zero = False)
plt.title("ACF of S&P500", size = 24)
plt.show()

# values on the x-axis represent lags, the y-axis is the correlation
# the blue area is the significance, coefficients should be larger than this area to be counted as significant
# the values are significantly different from 0 which suggests some auto correlation for that specific lag
# the blue area expands as lag values increase - ie the greater the distance in time, the more unlikely
# it is that the autocorrelation persists. (ie today's prices are most likely closer to yesterday's prices
# than prices from a month ago)

"""All the values in the autocorrelation graph are much higher than 1, which suggests a time dependence
in the data. Additionally, the autocorrelation barely diminishes as the lags increase, suggesting that
prices (even a month back) can still serve as decent estimators


Now lets do the autocorrelation of white noise"""
sgt.plot_acf(df_train["White noise"], lags = 40, zero = False)
plt.title("ACF of White noise", size = 24)
plt.ylim(-0.1, 0.1) # note how the coefficients are waaaaaay smaller
plt.show()

# there are patterns of positive and negative autocorrelation
# Note how almost all the lines fall within the significance area, thus the coefficients aren't 
# significant, suggesting that there isn't any auto correlation for any lag

sgt.plot_acf(df_train["rw"], lags = 40, zero = False)
plt.title("ACF of Random walk", size = 24)
#plt.ylim(-0.1, 0.1) # note how the coefficients are waaaaaay smaller
plt.show()

# note how the random walk ACF looks very similar to that of the data

"""Partial Autocorrelation

The autocorrelation coefficients calculated above show both the direct and indirect impacts each
lag has on the current value. Indirect refers to any other way this lag affects the current value, ie 
the fact that lag number three affects lag number two, which affects lag number one, which affects the
current value etc.

If we want to compute only the direct impact a lag has on the current value, we need to compute the
Partial Autocorrelation, which we can do using the Partial Autocorrelation Function (PACF)

"""
# Note: there are many ways of calculating the PACF, we're going to rely on the Order of Least Squares (OLS)
sgt.plot_pacf(df_train["Market value"], lags = 40, zero = False, method = ("ols"))
plt.title("PACF S&P500", size = 24)
plt.show()

# Note how most of the values are close to 0 after the first lag
# Some are even negative (like the 9th lag), which suggests that higher values 9 days ago create lower
# prices today
# The prices beyond the first three lags are generally not significant (ie they're in the blue)
# Note that values for the first lag in the ACF and PACF should be identical 

sgt.plot_pacf(df_train["White noise"], lags = 40, zero = False, method = ("ols"))
plt.ylim(-0.1, 0.1)
plt.title("PACF White noise", size = 24)
plt.show()

sgt.plot_pacf(df_train["rw"], lags = 40, zero = False, method = ("ols"))
plt.title("PACF Random walk", size = 24)
plt.show()


