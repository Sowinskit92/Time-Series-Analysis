#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:16:57 2024

@author: timsowinski
"""

import pandas as pd
import os
import requests
import json
import urllib.request
import sys
from datetime import datetime
from API_import1 import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.graphics.tsaplots as sgt



start_date = "2024-01-01"
end_date = "2024-12-31"

st = datetime.now()

Compile = False

if Compile == True:

    for i, j in enumerate([i for i in os.listdir() if "Wind-CCGT" in i]):
        
        df_temp = pd.read_csv(j)
        
        if i == 0:
            df = df_temp
        else:
            df = pd.concat([df, df_temp]).reset_index(drop = True)
    
    print(df)
    df.to_csv("Wind-CCGT generation data.csv", index = False)


#print(x)
#x = Data_Import().Elexon_query().Fuel_Mix_data(start_date, end_date, fuel_types = ["WIND", "CCGT"]).set_index("Start time")


def ACF_plot(data, title = "ACF plot", title_size = 18, lags = 40, ylim = 0.1, PACF = False):
    
    if PACF == False:
        sgt.plot_acf(data, zero = False, lags = lags)
    else:
        sgt.plot_pacf(data, zero = False, lags = lags)
    
    plt.title(title, size = title_size)
    plt.ylim(-1*ylim, ylim)
    plt.show()


# =============================================================================
# Loads training data
# =============================================================================
print("Loading data...")
MIP = pd.read_csv("MIP data.csv")
Dem = pd.read_csv("ITSDO data.csv")
gen_mix = pd.read_csv("Wind-CCGT generation data.csv")
gen_mix = pd.pivot_table(gen_mix, index = "Start time", values = "MW", columns = "Fuel type", aggfunc = "mean").reset_index()
"""really could do with nuclear/biomass too"""
# Need wind generation + the amount of wind generation bid off for each period (ie the total amount available to the market)
# Need capacity on outage too for exog variables

# =============================================================================
# Loads testing data
# =============================================================================

# Will need forecast capacity to come offline
# Will need forecast wind/solar generation
# Will need forecast demand
# Ideally also a forecast of gas prices


# =============================================================================
# Processing data
# =============================================================================
print("Processing data...")
df = MIP

# merges dataframes into one
for i in [Dem, gen_mix]:
    df = pd.merge(df, i, on = "Start time", how = "inner")


# removes unwanted columns
df = df[["Start time", "Price", "CCGT", "WIND", "Demand"]]
df = df.drop_duplicates(keep = "first")

df["Start time"] = pd.to_datetime(df["Start time"])
df["Start time"] = df["Start time"].dt.tz_localize(None)

df = df.set_index("Start time")

df = df.asfreq("30min")


# =============================================================================
# Checks for any entries which are blank
# =============================================================================
print("Checking data...")
blank_df = df[df["Price"].isna()]

if len(blank_df.index) > 0:
    #print(blank_df)
    df = df.bfill()
    #raise ImportError("There are blank prices in the starting dataframe. Please review")
else:
    pass

# removes blank_df from memory
del blank_df


# =============================================================================
# Creating test/training data
# =============================================================================
print("Creating training/testing datasets...")
train_from_date = datetime(2024, 1, 1, 0, 0, 0)
train_to_date = datetime(2024, 10, 31, 0, 0, 0)

df_train = df.loc[train_from_date: train_to_date]

#df_train = df[(df["Start time"].dt.date >= train_from_date.date()) & (df["Start time"].dt.date <= train_to_date.date())]
#df_train = df_train.set_index("Start time")


test_from_date = datetime(2024, 11, 1, 0, 0, 0)
test_to_date = datetime(2025, 1, 1, 0, 0, 0)

df_test = df.loc[test_from_date: test_to_date]

#df_test = df[(df["Start time"].dt.date >= test_from_date.date()) & (df["Start time"].dt.date < test_to_date.date())]
#df_test = df_test.set_index("Start time")



print((test_to_date - test_from_date).days*48)

print(len(df_test))

test_exog = df_test.loc[test_from_date: test_to_date, "WIND"]

print(len(test_exog.index))


# =============================================================================
# Building model
# =============================================================================
print("Fitting model...")
"""SARIMAX is likely to be the best one here"""

seasonal_periodicity = 48*365 # number of 30min periods in a year

# DOESN"T WORK YET model = SARIMAX(df_test["Price"], order = (1, 1, 1), seasonal_order = (1, 1, 1, 48), exog = df_test[["CCGT", "WIND", "Demand"]])
#model = SARIMAX(df_train["Price"], order = (1, 1, 1), exog = df_train[["WIND", "CCGT", "Demand"]])

results = model.fit()
resid = results.resid
summary = results.summary()

print(summary)
print(f"Model found in {str(datetime.now() - st)}s")
print("Predicting results...")

print(test_from_date, test_to_date)
prediction = results.predict(start = test_from_date, end = test_to_date - relativedelta(days = 1), exog = df_test[["WIND", "CCGT", "Demand"]].loc[test_from_date: test_to_date])

print(prediction)
print(df_test)
prediction.plot(color = "red", title = "ARMAX prediction")
df_test["Price"].plot(color = "blue")


print(f"Code complete in {str(datetime.now() - st)}s")


