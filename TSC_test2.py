# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:00:34 2025

@author: Tim.Sowinski
"""

# =============================================================================
# Currently requires python 3.10 to run, no higher
# =============================================================================
# EXAMPLE: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.RocketClassifier.html


from sktime.classification.kernel_based import RocketClassifier
from CI_data_imports20 import *
pd.set_option("display.max_columns", None)
# =============================================================================
# I'll take a number of units which we confidently know are:
#   - batteries
#   - peakers
#   - solar
#   - CCGTs
#   - biomass
#
# I'll then train the model on each of these
# =============================================================================

# =============================================================================
# Outlining the units by fuel type which will be needed to train the model.
# Note, these are just some for testing. In reality I should use all of the 
# ones we know
# =============================================================================
solar_units: list = ["T_SUTBS-1", "T_LARKS-1"]
peaking_units: list = ["E_PETEM-3", "E_PGLLAN", "E_GOSHS-1", "E_SOLUTIA"]
bess_units: list = ["E_DOLLB-1"]
CCGT_units: list = ["T_PEHE-1", "T_SEAB-2", "T_CNQPS-1", "T_MRWD-1", "T_KEAD-2"]
"""
BMUs = Data_Import().SQL_query().BMU_data()

CCGT_units = BMUs[BMUs["Fuel type"] == "CCGT"]["BMU ID"].unique().tolist()
bess_units = BMUs[(BMUs["Fuel type"] == "CCGT") & (~BMUs["BMU ID"].str.contains("__"))] # removes 2__, V__ units as these can change fuel types
peaking_units = BMUs[(BMUs["Fuel type"] == "Gas Recip") & (~BMUs["BMU ID"].str.contains("__"))]
solar_units = BMUs[(BMUs["Fuel type"] == "Solar") & (~BMUs["BMU ID"].str.contains("__"))]
"""

# =============================================================================
# Loads and sorts data - time series must be increasing
# =============================================================================

df = pd.read_pickle("PN_train_data2.pickle")
df = df.sort_values(by = "TimeTo")


fts = df["ft"].unique().tolist()
ft_int_dict = {j: i for i, j in enumerate(fts)}
int_ft_dict = {j: i for i, j in ft_int_dict.items()}
df["ft_int"] = df["ft"].map(ft_int_dict)

# =============================================================================
# Sets up training data
# =============================================================================

df = df.set_index(["ft_int", "TimeTo"])[["LevelTo"]]

df_train = df.iloc[:int(len(df.index)*0.8)]
df_test = df.iloc[int(len(df.index)*0.8):]

X_train = df_train
Y_train = pd.DataFrame(index = [i for i in range(len(fts))])
Y_train["ft"] = Y_test.index.map(int_ft_dict)

print("Fitting ROCKET model...")
clf = RocketClassifier(num_kernels=512) 
clf.fit(X_train, Y_train) 

# this works but there are unequal length series which it can't deal with
# I think this is because batteries tend to submit more than one per SP

# To deal with this, it would probably make most sense to go down to minute granularity
# and just backfill the data if there's nothing there for that minute

print("Predicting...")
y_pred = predict(x_test)






