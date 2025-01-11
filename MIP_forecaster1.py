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
from statsmodels.tsa.statespace.sarimax import SARIMAX

start_date = "2024-01-01"
end_date = "2024-12-31"

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


# =============================================================================
# Checks for any entries which are blank
# =============================================================================
print("Checking data...")
blank_df = MIP[MIP["Price"].isna()]

if len(blank_df.index) > 0:
    raise ImportError("There are blank prices in the starting dataframe. Please review")
else:
    pass

# removes blank_df from memory
del blank_df


# =============================================================================
# Creating test/training data
# =============================================================================
print("Creating training/testing datasets...")




# =============================================================================
# Building model
# =============================================================================
print("Fitting model...")
"""SARIMAX is likely to be the best one here"""

seasonal_periodicity = 48*365 # number of 30min periods in a year

model = SARIMAX(df["Price"], order = (1, 1, 1), seasonal_order = (2, 0, 1, seasonal_periodicity), exog = df[["CCGT", "WIND", "Demand"]])
results = model.fit()
resid = results.resid
summary = results.summary()
print(summary)
print(results.params.round(3))








