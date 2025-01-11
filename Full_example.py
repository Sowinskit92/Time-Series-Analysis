#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 08:58:35 2025

@author: timsowinski
"""

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from pmdarima.arima import OCSBTest
from arch import arch_model
import yfinance
import warnings
from sktime.forecasting.arima import AutoARIMA
warnings.filterwarnings("ignore")
sns.set()


# =============================================================================
# This example is going over Volkswagen's take over of Porsche and the effect
# of various events on the performance of the models
# =============================================================================

# we use BMW as a market benchmark
raw_data = yfinance.download(tickers = "VOW3.DE, PAH3.DE, BMW.DE", interval = "1d", group_by = "ticker",
                             auto_adjust = True, threads = True)

df = raw_data.copy()

# =============================================================================
# Defining the important dates
# =============================================================================
start_date = "2009-04-05" # earliest date we have data for both stocks
ann_1 = "2009-12-09" # 1st announcement from Volkswagen that they now own 49.9% of Porsche
ann_2 = "2012-07-05" # 2nd announcement from Volkswagen that they had bought the remaining 50.1% of Porsche
# note: it's likely that beyond this point the values for Volkswagen and Porsche will move similarly
end_date = "2014-01-01" # arbitrary end date
diesel_gate = "2015-09-20" # day diesel-gate came out
diesel_gate_y1 = "2016-09-20" # diesel-gate + 1yr

# creating dataframe of closing values
df["vol"] = df["VOW3.DE"].Close
df["por"] = df["PAH3.DE"].Close
df["bmw"] = df["BMW.DE"].Close

# adding returns
df["ret_vol"] = df["vol"].pct_change(1).mul(100)
df["ret_por"] = df["por"].pct_change(1).mul(100)
df["ret_bmw"] = df["bmw"].pct_change(1).mul(100)

# creating squared returns for the volatility calculations down the line
df["sq_vol"] = df.ret_vol.mul(df.ret_vol)
df["sq_por"] = df.ret_por.mul(df.ret_por)
df["sq_bmw"] = df.ret_bmw.mul(df.ret_bmw)

# separating the volume of traded stock, as the more volume traded, the more likely the value to fluctuate
df["q_vol"] = df["VOW3.DE"].Volume
df["q_por"] = df["PAH3.DE"].Volume
df["q_bmw"] = df["BMW.DE"].Volume

# setting frequency to business days
df = df.asfreq("b")

# sorting missing values
df = df.bfill()

# removing now unnecessary data
del df["VOW3.DE"], df["PAH3.DE"], df["BMW.DE"]

"""
Note: we aren't splitting it up into a training and testing set as we aren't forecasting here,
we're just examining a specific point in time
"""

# =============================================================================
# Plotting the data
# =============================================================================

# used to graph series as a specific colour
colour_dict = {"vol": "blue", "por": "green", "bmw": "gold"}

for series, colour in colour_dict.items():
    df[series].plot(color = colour, title = "Value of Car Brands")
    plt.legend([i for i in colour_dict.keys()])
plt.show()
# there are some broad correlation in how they move, indicating wider trends of the car industry
# but VW's moves a lot more than the others in ~2010-2015

# =============================================================================
# Adding different colours to the chart to make it clearer to see
# =============================================================================


df["vol"][start_date: ann_1].plot(color = "#33B8FF") 
df["por"][start_date: ann_1].plot(color = "#49FF3A")
df["bmw"][start_date: ann_1].plot(color = "#FEB628")

df["vol"][ann_1: ann_2].plot(color = "#1E7EB2") 
df["por"][ann_1: ann_2].plot(color = "#2FAB25")
df["bmw"][ann_1: ann_2].plot(color = "#BA861F")

df["vol"][ann_2: end_date].plot(color = "#0E3A52") 
df["por"][ann_2: end_date].plot(color = "#225414")
df["bmw"][ann_2: end_date].plot(color = "#7C5913")

df["vol"][end_date: diesel_gate].plot(color = "blue") 
df["por"][end_date: diesel_gate].plot(color = "green")
df["bmw"][end_date: diesel_gate].plot(color = "gold")

df["vol"][diesel_gate: diesel_gate_y1].plot(color = "blue", linestyle = "dashed") 
df["por"][diesel_gate: diesel_gate_y1].plot(color = "green", linestyle = "dashed")
df["bmw"][diesel_gate: diesel_gate_y1].plot(color = "gold", linestyle = "dashed")

plt.legend([i for i in colour_dict.keys()])
plt.show()

# it appears as though BMW and Porsche a much more similar as time series than Volkswagen
# to quantify this, we can look at their correlation

# =============================================================================
# Checking correlation between the different time series
# =============================================================================

brand_correl = {"vol1": "por", "vol2": "bmw", "por": "bmw"}
'''
Commented out so it doesn't keep running unnecessarily

print(f"Correlation between car manufacturers between {start_date} and {end_date}:")

brand_correl = {"vol1": "por", "vol2": "bmw", "por": "bmw"}
for i, j in brand_correl.items():
    if "vol" in i:
        i = "vol"
    
    print(f"{i} & {j} correlation:")
    correl = df[i][start_date:end_date].corr(df[j][start_date:end_date])
    print(f"\t{correl:.2f}")
'''
# what we see is that Volkswagen moves in a much more similar way to the market benchmark (BMW)
# but from the graph we know it just does so at much higher value

# however this doesn't make too much sense, as we'd expect Volkswagen and Porsche to move much 
# more similarly after the take over. So really, we need to examine each time period individually

d1 = diesel_gate
d2 = diesel_gate_y1
print(f"Correlation between car manufacturers between {d1} and {d2}:")
for i, j in brand_correl.items():
    if "vol" in i:
        i = "vol"
    
    print(f"{i} & {j} correlation:")
    if d2 == "":
        correl = df[i][d1:].corr(df[j][d1:])
    else:
        correl = df[i][d1: d2].corr(df[j][d1: d2])
    print(f"\t{correl:.2f}")

# we see much lower correlations between start_date and ann_1, but BMW and VW are most closely 
# correlated with Porsche and BMW being the least correlated. This suggests the stock prices for BMW
# and Porsche didn't behave too similarly before the buyout

# A possible explanation for what's going on is that after the buyout, Porsche's evaluation becomes directly
# tied to that of VW's. VW then becomes a market trend setter, which indirectly affects BMW's valuation
# (and all other European car brands).
# Ie this makes it appear as though Porsche and BMW's valuations behave similarly after the buyout
# without a direct link between them

# If this is the case, we'd expect that the BMW/Porsche correlation would grow as VW becomes even more
# of a market leader. We'd also expect the VW/Porsche correlation to grow as VW buys more of Porsche

# We can test this by changing the d1 and d2 values above


# Running the correlations between d1 = ann_1 and d2 = ann_2, we get a VW/BMW correlation of 98%, as
# we all as higher correlations between VW/Porsche and Porsche/BMW

# Running the correlations between d1 = ann_2 and d2 = end_date, we see very high correlations between 
# them all, but interestingly, the highest correlation is between BMW and Porsche

# Running the correlation from d1 = end_date onwards for the rest of the dataset, you find that 
# Porsche and VW have a strong correlation of ~85% but Porsche and BMW aren't very correlated at all (~25%)
# VW/BMW are only moderately correlated (~50%)

# The reason comes down to the diesel gate scandal. Up to this date they're very very correlated (80%+)
# but in the year after, BMW is only ~20% correlated, indicating that VW was hit much harder than BMW


# =============================================================================
# Best fitting models
# =============================================================================

"""
Models are supposed to capture the current trends in the market, but if the trends shift then so
should the model
"""

'''
Commenting out to avoid it running unnecessarily

# checking the best model for VW across three of the time periods:

# note there is an arguement that you shouldn't include the exogenous variables in this case due
# to their very high correlation, as this will take away from the explanitory power of the lagged VW prices
mod_pr_pre_vol = auto_arima(df.vol[start_date: ann_1], exogenous = df[["por", "bmw"]][start_date: ann_1], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_pre_vol.summary())
mod_pr_btn_vol = auto_arima(df.vol[ann_1: ann_2], exogenous = df[["por", "bmw"]][ann_1: ann_2], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_btn_vol.summary())
mod_pr_post_vol = auto_arima(df.vol[ann_2: end_date], exogenous = df[["por", "bmw"]][ann_2: end_date], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_post_vol.summary())

# you can see that the best fitting model changes as circumstances of the market change
# ie new information can have a high impact on trends compared to past price movements

# Doing the same for Porsche to see if the trend presists

mod_pr_pre_por = auto_arima(df.por[start_date: ann_1], exogenous = df[["vol", "bmw"]][start_date: ann_1], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_pre_por.summary())
mod_pr_btn_por = auto_arima(df.por[ann_1: ann_2], exogenous = df[["vol", "bmw"]][ann_1: ann_2], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_btn_por.summary())
mod_pr_post_por = auto_arima(df.por[ann_2: end_date], exogenous = df[["vol", "bmw"]][ann_2: end_date], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_pr_post_por.summary())

# similar trends occur for Porsche, backing up the idea that current events are a bigger driver of price
# compared to previous price movements


# doing the same but for returns instead of prices

mod_ret_pre_vol = auto_arima(df.ret_vol[start_date: ann_1], exogenous = df[["ret_por", "ret_bmw"]][start_date: ann_1], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_pre_vol.summary())
mod_ret_btn_vol = auto_arima(df.ret_vol[ann_1: ann_2], exogenous = df[["ret_por", "ret_bmw"]][ann_1: ann_2], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_btn_vol.summary())
mod_ret_post_vol = auto_arima(df.ret_vol[ann_2: end_date], exogenous = df[["ret_por", "ret_bmw"]][ann_2: end_date], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_post_vol.summary())


mod_ret_pre_por = auto_arima(df.ret_por[start_date: ann_1], exogenous = df[["ret_vol", "ret_bmw"]][start_date: ann_1], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_pre_por.summary())
mod_ret_btn_por = auto_arima(df.ret_por[ann_1: ann_2], exogenous = df[["ret_vol", "ret_bmw"]][ann_1: ann_2], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_btn_por.summary())
mod_ret_post_por = auto_arima(df.ret_por[ann_2: end_date], exogenous = df[["ret_vol", "ret_bmw"]][ann_2: end_date], 
                            m = 5, max_p = 5, max_q = 5)
print(mod_ret_post_por.summary())
'''

# =============================================================================
# Forecasting the best fitting models into the future
# =============================================================================

# we're going to use the VW value after the first announcement, but aren't including the other prices as exogenous variables
'''
model_auto_pred_pr = auto_arima(df.vol[start_date: ann_1], m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5, trend = "ct")

# storing the results
df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods = len(df[ann_1:ann_2])), index = df[ann_1:ann_2].index)
df_auto_pred_pr[ann_1:ann_2].plot(color = "red")
df.vol[ann_1:ann_2].plot(color = "blue")
plt.show()
"""
We see that the prediction does okay to start with but does terribly after a few months, with prices actually going negative.
To see what's going on we can look at the graph in more detail
"""

# over the first few months, the model captures the general trend but not the values day by day. So predictions are good for a ballpark estimate initially
df_auto_pred_pr[ann_1: "2010-03-01"].plot(color = "red")
df.vol[ann_1: "2010-03-01"].plot(color = "blue")
plt.show()
'''
# lets see what happens if the exogenous variables are added
X_test = df["por"][start_date: ann_1]
print(X_test)
"""
# It's the X variable which is the problem. Guidance notes says it takes a dataframe but it doesn't, it just fails. It doesn't like a 2d array either
# I can't find anything else online about it with this new syntax
# The error it gives is unhelpful too. It tells you to reshape the data but then fails when you do it as it tells you to.
# In fact one of them tells you to turn it into a dataframe! But the dataframe doesn't work either 

model_auto_pred_pr = auto_arima(df.vol[start_date: ann_1], X = X_test,
                                m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5, trend = "ct")

# storing the results
df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods = len(df[ann_1:ann_2]), X = df["por"][ann_1: ann_2].values),
                               index = df[ann_1:ann_2].index)


df_auto_pred_pr[ann_1:ann_2].plot(color = "red")
df.vol[ann_1:ann_2].plot(color = "blue")
plt.show()

# if the exog function worked like it should do, this would result in a much better model than before. Even better if the market benchmark (BMW) is used
# and it's even better if we use 
"""

"""I've now found sktime's autoarima and will try that instead"""
"""Is a bust, it doesn't have an option for exog variables in the model and also uses the pmdautoarima to work anyway"""

#model_auto_pred_pr = AutoARIMA(df.vol[start_date: ann_1], max_p = 5, max_q = 5, max_P = 5, max_Q = 5, trend = "ct")


# =============================================================================
# Checking the volatility
# =============================================================================


df["sq_vol"][start_date:ann_1].plot(color = "#33B8FF")
df["sq_vol"][ann_1:ann_2].plot(color = "#1E7EB2")
df["sq_vol"][ann_2:end_date].plot(color = "#0E3A52")

# you can see there's a huge amount of volatility leading up to both announcements of VW buying some of Porsche but it settles afterwards

GARCH_pre = arch_model(df.ret_vol[start_date: ann_1], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_GARCH_pre = GARCH_pre.fit(update_freq = 5)
print(results_GARCH_pre.summary())


GARCH_btn = arch_model(df.ret_vol[ann_1: ann_2], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_GARCH_btn = GARCH_pre.fit(update_freq = 5)
print(results_GARCH_btn.summary())

GARCH_post = arch_model(df.ret_vol[ann_2: end_date], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_GARCH_post = GARCH_pre.fit(update_freq = 5)
print(results_GARCH_post.summary())



















