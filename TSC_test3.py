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
import math
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

def to_minute_granularity(df):
    # =============================================================================
    # This function is to take a time series df and convert it to one of 
    # minute granularity  
    # =============================================================================
    
    time_col = "TimeTo"
    unit_col = "BMU ID"
    target_col = "LevelTo"
    
    units = df[f"{unit_col}"].unique().tolist()
    
    # below creates a time series going between the min and max dates of the PN data submitted
    max_time = df[f"{time_col}"].max()
    min_time = df[f"{time_col}"].min()
    
    ts = pd.date_range(start = str(min_time), end = str(max_time), freq = "1min")
    #print(df)
    
    for i, unit in enumerate(units):
        #print(unit)
        dict_temp = df[df["BMU ID"] == unit].set_index(f"{time_col}").to_dict()[f"{target_col}"]
        
        ft_temp = df[df["BMU ID"] == unit]["ft"].unique().tolist()
        print(ft_temp)
        
        if len(ft_temp) > 1:
            raise ImportError(f"{unit} has multiple fuel types attributed to it. Please review and try again")
        else:
            ft_temp = ft_temp[0]
        
        
        if i == 0:
            df_new = pd.DataFrame(index = ts)
            df_new["ft"] = ft_temp
            df_new[f"{target_col}"] = df_new.index.map(dict_temp)
            
            # sets first value to 0 if there isn't one (the longer the dataset, the less of a problem this will be)
            if math.isnan(df_new.iloc[0][f"{target_col}"]):
                df_new.iloc[0] = 0
            else:
                pass
            
            df_new[f"{target_col}"] = df_new[f"{target_col}"].ffill()
            #df_new[f"{unit_col}"] = unit
            
        else:
            df_temp = pd.DataFrame(index = ts)
            df_temp["ft"] = ft_temp
            df_temp[f"{target_col}"] = df_temp.index.map(dict_temp)
            
            # sets first value to 0 if there isn't one (the longer the dataset, the less of a problem this will be)
            if math.isnan(df_temp.iloc[0][f"{target_col}"]):
                df_temp.iloc[0] = 0
            else:
                pass
            
            df_temp[f"{target_col}"] = df_temp[f"{target_col}"].ffill()
            
            #df_temp[f"{unit_col}"] = unit
            df_new = pd.concat([df_new, df_temp])
    
    
    
    df_new = df_new.reset_index()
    df_new = df_new.rename(columns = {"index": "TimeTo"})
    
    # checks for NA rows
    #print(df_new[df_new.isna().any(axis = 1)])
    
    return df_new


# =============================================================================
# Loads and sorts data - time series must be increasing
# =============================================================================

df = pd.read_pickle("PN_train_data2.pickle")
df = df.sort_values(by = "TimeTo")

df = to_minute_granularity(df)
print(df)

fts = df["ft"].unique().tolist()
print(fts)
raise ImportError("For some reason there's a random 0 in here?")
sys.exit()
ft_int_dict = {j: i for i, j in enumerate(fts)}
print(ft_int_dict)
int_ft_dict = {j: i for i, j in ft_int_dict.items()}
df["ft_int"] = df["ft"].map(ft_int_dict)
print(df)
# =============================================================================
# Sets up training data
# =============================================================================

df = df.sort_values(by = "TimeTo")
df = df.set_index(["ft_int", "TimeTo"])[["LevelTo"]]

na_check = df[df.isna().any(axis = 1)]

if len(na_check.index) > 0:
    print(na_check)
    raise ImportError("There are NA values in the dataset, please review and try again")

df_train = df.iloc[:int(len(df.index)*0.8)]
df_test = df.iloc[int(len(df.index)*0.8):]

X_train = df_train
Y_train = pd.DataFrame(index = [i for i in range(len(fts))])
Y_train["ft"] = Y_test.index.map(int_ft_dict)
print(X_train)

test = df.reset_index()
for i in test["ft_int"].unique().tolist():
    print(len(test[test["ft_int"] == i].index))
print(df[df.isnull()])
print("How are there NA values? They should have all been gotten rid of?")
sys.exit()

print("Fitting ROCKET model...")
clf = RocketClassifier(num_kernels=512) 
clf.fit(X_train, Y_train) 

# this works but there are unequal length series which it can't deal with
# I think this is because batteries tend to submit more than one per SP

# To deal with this, it would probably make most sense to go down to minute granularity
# and just backfill the data if there's nothing there for that minute

print("Predicting...")
y_pred = predict(x_test)






