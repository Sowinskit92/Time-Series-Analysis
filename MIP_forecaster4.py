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
from API_import3 import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.graphics.tsaplots as sgt
from pmdarima.arima import auto_arima

# =============================================================================
# This version takes into account unavailable capacity of technology types
# =============================================================================

start_date = "2024-01-01"
end_date = "2024-12-31"

st = datetime.now()

pd.set_option("display.max_columns", None)


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

#print(x)
#x = Data_Import().Elexon_query().REMIT(start_date, end_date, only_last_revision = True)

#x.to_csv("REMIT data 2024 new.csv", index = False)




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
BMU_info = pd.read_csv("BMU Info.csv")
MIP = pd.read_csv("MIP data.csv")
Dem = pd.read_csv("ITSDO data.csv")
gen_mix = pd.read_csv("Wind-CCGT generation data.csv")
gen_mix = pd.pivot_table(gen_mix, index = "Start time", values = "MW", columns = "Fuel type", aggfunc = "mean").reset_index()

REMIT = pd.read_csv("REMIT data 2024.csv") # unavailable capacity by fuel type
REMIT = REMIT[REMIT["Status"] == "Active"]
REMIT["Start time"] = pd.to_datetime(REMIT["Start time"]).dt.tz_localize(None)
REMIT["End time"] = pd.to_datetime(REMIT["End time"]).dt.tz_localize(None)
"""really could do with nuclear/biomass/solar too"""
# Need the amount of wind generation bid off for each period (ie the total amount available to the market)

# =============================================================================
# Loads testing data
# =============================================================================

# Will need forecast capacity to come offline
# Will need forecast wind/solar/CCGT generation
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


# below adds the unavailable capacity by fuel type
df["End time"] = df["Start time"] + pd.DateOffset(minutes = 30)

d_range = df["Start time"].unique().tolist()

fuel_types_to_use = ["CCGT", "Interconnector", "NUCLEAR", "BIOMASS", "PS"]


"""
I need to do some more research into this as I'm getting some odd numbers as to the amount of 
unavailable capacity by fuel type, even when filtering out the dismissed ones and setting last_revision_only to tru
"""
for i in fuel_types_to_use:
    # iterates over the fuel types to add their unavailable MW
    print(i)
    REMIT_by_fuel_type = REMIT[REMIT["Fuel type"] == i]
    """Lambda function here??"""
    
    df[f"Unavailable {i}"] = [REMIT_by_fuel_type[(REMIT_by_fuel_type["Start time"] <= date) & (REMIT_by_fuel_type["End time"] >= date)]["Unavailable MW"].sum() for date in d_range]
    """
    #df[f"Unavailable {i}"] = df.apply(lambda x: REMIT_by_fuel_type[(REMIT_by_fuel_type["Start time"] <= x["Start time"])]["Unavailable MW"].sum())
    sys.exit()
    
    df[f"Unavailable {i}"] = REMIT_by_fuel_type["Unavailable MW"].sum().where((REMIT_by_fuel_type["Start time"] <= df["Start time"]) &
                                                                              REMIT_by_fuel_type["End time"] >= df["End time"])
    """


print(df)
sys.exit()




# creates final dataframe
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


test_from_date = datetime(2024, 11, 1, 0, 0, 0)
test_to_date = datetime(2025, 1, 1, 0, 0, 0)

df_test = df.loc[test_from_date: test_to_date]



# =============================================================================
# Building model
# =============================================================================
t = datetime.now()
print(f"Model fitting beginning at {t}...")
"""SARIMAX is likely to be the best one here"""

seasonal_periodicity = 48*365 # number of 30min periods in a year

# DOESN"T WORK YET model = SARIMAX(df_test["Price"], order = (1, 1, 1), seasonal_order = (1, 1, 1, 48), exog = df_test[["CCGT", "WIND", "Demand"]])
"""
This one works, I'm just trying the auto_arima too
model = SARIMAX(df_train["Price"], order = (1, 1, 1), exog = df_train[["WIND", "CCGT", "Demand"]])
results = model.fit()
resid = results.resid
summary = results.summary()

print(summary)

print(test_from_date, test_to_date)
prediction = results.predict(start = test_from_date, end = test_to_date - relativedelta(days = 1), exog = df_test[["WIND", "CCGT", "Demand"]].loc[test_from_date: test_to_date])

print(prediction)
"""

model_auto = auto_arima(df_train["Price"], X = df_train[["WIND", "CCGT", "Demand"]], m = 4, max_p = 5, 
                        max_q = 5, max_P = 3, max_Q = 3)

# AUTO_ARIMA finds a SARIMAX(2, 1, 3)(1, 0, 1, 4)

print(model_auto.summary())
print(f"Model found in {str(datetime.now() - st)}s")

prediction_auto = pd.DataFrame(model_auto.predict(n_periods = len(df_test.loc[test_from_date: test_to_date].index),
                                             X = df_test[["WIND", "CCGT", "Demand"]]),
                          index = df_test[test_from_date:test_to_date].index)

"""
This was the original Auto_ARIMA thing I had that wasn't working
prediction = pd.DataFrame(model.predict(n_periods = len(df_test.loc[test_from_date: test_to_date].index),
                                        X = df_train[["WIND", "CCGT", "Demand"]][test_from_date:test_to_date]),
                            index = df_test[test_from_date:test_to_date].index)
"""
print(prediction_auto)

print(f"Model found in {str(datetime.now() - st)}s")
print("Predicting results...")


prediction_auto.plot(color = "red")
df_test["Price"].plot(color = "blue")

print(f"Code complete in {str(datetime.now() - st)}s")


