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
#   - wind
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
    print(len(ts))
    print(max_time)
    print(min_time)
    sys.exit()
    
    #print(df)
    
    for i, unit in enumerate(units):
        #print(unit)
        dict_temp = df[df["BMU ID"] == unit].set_index(f"{time_col}").to_dict()[f"{target_col}"]
        
        ft_temp = df[df["BMU ID"] == unit]["ft"].unique().tolist()
        #print(ft_temp)
        
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
            
            #print(df_new[df_new["ft"] == 0])
            
            df_new[f"{target_col}"] = df_new[f"{target_col}"].ffill()
            #df_new[f"{unit_col}"] = unit
            #print(df_new)
            
        else:
            df_temp = pd.DataFrame(index = ts)
            df_temp["ft"] = ft_temp
            
            
            df_temp[f"{target_col}"] = df_temp.index.map(dict_temp)
            
            
            # sets first value to 0 if there isn't one (the longer the dataset, the less of a problem this will be)
            if math.isnan(df_temp.iloc[0][f"{target_col}"]):
                df_temp.iloc[0, df_temp.columns.get_loc(f"{target_col}")] = 0
            else:
                pass
            
            df_temp[f"{target_col}"] = df_temp[f"{target_col}"].ffill()
            
            #df_temp[f"{unit_col}"] = unit
            #print(df_new)
            df_new = pd.concat([df_new, df_temp])
            #print(df_new[df_new["ft"] == 0])
    
    df_new = df_new.reset_index()
    df_new = df_new.rename(columns = {"index": "TimeTo"})
    
    # checks for NA rows
    #print(df_new[df_new.isna().any(axis = 1)])
    
    return df_new


# =============================================================================
# Loads and sorts data - time series must be increasing
# =============================================================================

df = pd.read_pickle("PN_train_data3.pickle")

df = df.sort_values(by = "TimeTo")

df = to_minute_granularity(df)


fts = df["ft"].unique().tolist()

# This was coming from a coding error setting the entire first row to 0 if it was empty
# rather than just the LevelTo column
#raise ImportError("For some reason there's a random 0 in here?")

ft_int_dict = {j: i for i, j in enumerate(fts)}
print(ft_int_dict)
int_ft_dict = {j: i for i, j in ft_int_dict.items()}
df["ft_int"] = df["ft"].map(ft_int_dict)

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
Y_train["ft"] = Y_train.index.map(int_ft_dict)
print(X_train)
print(Y_train)


print("Fitting ROCKET model...")
clf = RocketClassifier(num_kernels=512) 
clf.fit(X_train, Y_train) 

# after all the changing, I'm still getting the same error:
# ValueError: Data seen by RocketClassifier instance has unequal length series, but this RocketClassifier instance cannot handle unequal length series. Calls with unequal length series may result in error or unreliable results.
# I think this could be because I'm using two solar units and 1 battery unit - maybe I should try with 2 battery units (but this isn't ideal as we don't have equal numbers of units for the real model to use)
# Potentially another option could be to add the BMU ID to make it a multi-variate model? I don't know if this will fix it but it could be an option


# I've just tried adding another battery unit and it does get further than it did without it. So we need equal numbers of fuel types to run this model annoyingly
# I now get this error: ValueError: cannot reshape array of size 258_144 into shape (2,64_536,1)
# 258144 is the number of rows in the training dataset
# 2 is the number of fuel types (I assume?)
# 1 is the number of variables (LevelTo)??? (No idea tbh)
# Absolutely no clue where the 64_536 is coming from though

# I found a post saying that for this to work, the number needs to match num_rows*num_cols 
# https://stackoverflow.com/questions/42947298/valueerror-cannot-reshape-array-of-size-30470400-into-shape-50-1104-104
# with this 64_536 = 258_144/4 - not sure why it's 4 though? You'd think it'd be 2?

# the error I get also says: X_values.reshape(n_instances, n_timepoints, n_columns) is where it's going wrong
# so it thinks I've got 64_536 time points, but this should be 80_670 (len of ts date_range)

# Oh crap yeah! The min/max date range is 2025-01-01 00:01:00 - 2025-02-26 00:30:00
# BUT - it looks like the time value only goes up to 2025-02-14 19:36:00 for some units

# Ahh no wait, that's because the df was showing X_train (which is 80% the length of ts)

# Right, 64_536 = 80_670*0.8 (ie the length of ts multipled by 0.8 to get the X_train df)
# So somehow, I need to get 64_536 doubled to 129_072 OR 258_144 halved. But how???

# ITS BECAUSE THERE's TWO TIME SERIES PER UNIT TYPE AT THE MOMENT (ie the test data I've used has 2 BESS units and 2 solar units, but there's no distinction between them atm)
# So adding the unit as an additional index should work right? As the n_timepoints would remain 64_536 but there would be another column showing the unit which would make them equal 258_144 (hopefully...?)


print("Predicting...")
y_pred = clf.predict(x_test)






