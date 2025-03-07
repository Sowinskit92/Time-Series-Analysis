# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:32:49 2025

@author: Tim.Sowinski
"""

from CI_data_imports21 import *


BMUs = Data_Import().SQL_query().BMU_data()
known_peakers = BMUs[(BMUs["Fuel type"].astype(str).str.lower().str.contains("recip")) & (~BMUs["BMU ID"].astype(str).str.startswith("2__"))]["BMU ID"].unique().tolist()
known_bess = BMUs[(BMUs["Fuel type"].astype(str).str.lower().str.contains("battery")) & (~BMUs["BMU ID"].astype(str).str.startswith("2__"))]["BMU ID"].unique().tolist()
known_solar = BMUs[(BMUs["Fuel type"].astype(str).str.lower().str.contains("solar")) & (~BMUs["BMU ID"].astype(str).str.startswith("2__"))]["BMU ID"].unique().tolist()
known_demand = BMUs[(BMUs["Fuel type"].astype(str).str.lower().str.contains("demand")) & (~BMUs["BMU ID"].astype(str).str.startswith("2__"))]["BMU ID"].unique().tolist()

date_from = "2023-01-01"
date_to = "2025-03-06"

file_name = "Known unit FPNs.pickle"

if file_name not in os.listdir():
    known_units = known_peakers + known_bess + known_solar + known_demand
    df = Data_Import().SQL_query().PN_data(date_from = date_from, date_to = date_to, BMU_IDs = known_units)
    df.to_pickle(file_name)
else:
    df = pd.read_pickle(file_name)

# =============================================================================
# Potential ideas:
#   1. Look at sum/mean LevelTo values by SP (doesn't give much tbh)
#   2. Look at sum/mean LevelTo values by SP as a % of the max FPN for the day (I'm not having much luck with this either)
#   3. Look at the running profile relative to the MIP by SP + gas price for the day
# =============================================================================



# =============================================================================
# Idea 1
# =============================================================================
#df = df.groupby("HHPeriod")["LevelTo"].mean()
#print(df)


# =============================================================================
# Idea 2 - I can't get this working for BESS units for some reason
# =============================================================================
print(df)
df = df[df["ft"] == "Gas Recip"]

df["Date"] = df["TimeTo"].dt.date

max_FPN = df.groupby(["Date", "BMU ID"])["LevelTo"].max().reset_index().rename(columns = {"LevelTo": "max_FPN"})

print(max_FPN[max_FPN["max_FPN"] != 0].sort_values(by = ["Date", "BMU ID"]))


df = pd.merge(df, max_FPN, on = ["Date", "BMU ID"], how = "outer")

df["max_FPN"] = df["max_FPN"].abs()

# looks like like merge has been done correctly
print(df[df["max_FPN"] != 0].sort_values(by = ["Date", "BMU ID"]))

# fraction of output
df["frac"] = (df["LevelTo"].div(df["max_FPN"]))

# I tried adding a square to make the differences more pronounced
#df["frac"] = (df["LevelTo"].div(df["max_FPN"]))**2
# I tried moving the squared to try and make a clearer cut through between smaller and larger values. Didn't really help though
#df["frac"] = (df["LevelTo"]**2).div(df["max_FPN"]).abs()

# removes any values where the max_FPN for the day was 0, as this just comes through as 0 in the mean
#df = df[df["max_FPN"] != 0]
print(df)

results = df.groupby("HHPeriod")["frac"].mean()
#results = results**0.5
print(results)

# =============================================================================
# Given these average SP profiles which the above code spits out, how would you
# be able to say that a SP profile of another unit matches that?
# My idea is that you could maybe sum up the results values and have a error margin
# away from this it would be classed as. 
# ----> But how would you quantify this? Just doing a normal sum wouldn't take
# ----> into account the periods of generation, meaning a unit could be falsely
# ----> classified if it generated in a way which made the value similar enough
# ----> even though the profile doesn't match a peaker.
#
# ----> You couldn't just weight against the SP number either because later SPs
# ----> will count for more, meaning the same thing could happen
# ----> 
#
# HOL' UP - actually maybe you could do this. 
# ----> If you ran the above code for one day, you could see the score's of each
# ----> of the known peakers. You could then compare this to units which are unknown
# ----> and see which ones score similarly - this would inherently take into account
# ----> the MIP/gas prices of the periods in question
# =============================================================================




