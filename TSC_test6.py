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
from CI_data_imports21 import *
import math
from sktime.utils import mlflow_sktime

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


# =============================================================================
# Model1 is trained on just E_DOLLB-1 (BESS), T_LARKS-1 (solar), and E_GOSHS-1 (peaker)
# Between 2024-01-01 and 2025-03-06 using straight PNs (doesn't use fraction of max PN by day yet)
# Ideally, I'd be able to incorporate more than just one unit into the testing 
# data and potentially use the fraction of the max submitted PN by day
# =============================================================================

# =============================================================================
# This still doesn't work with multiple units per fuel type
# I think I'll need to add the IDs as a feature??
# =============================================================================


Model_Name = "TSC_model1.json"

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

def Load_PN_data(date_from: str, date_to: str, units: list, file_name: str):
    
    if file_name.endswith(".pickle"):
        pass
    else:
        raise ImportError("Please enter file_name as a .pickle file")
    
    if file_name not in os.listdir():
    
        # =============================================================================
        # Gathers PN data to export to pickle file of specified units
        # =============================================================================
        
        BMUs = Data_Import().SQL_query().BMU_data()
        BMUs = BMUs[BMUs["BMU ID"].isin(units)]
        
        Unit_IDs = BMUs["BMUnitID"].astype(int).unique().tolist()
        Unit_IDs = tuple(Unit_IDs) # converts list to tuple to use in SQL query
        
        
        
        BMU_to_ID = BMUs.set_index("BMU ID").to_dict()["BMUnitID"]
        ID_to_BMU = BMUs.set_index("BMUnitID").to_dict()["BMU ID"]
        BMU_to_ft = BMUs.set_index("BMU ID").to_dict()["Fuel type"]
        ID_to_ft = BMUs.set_index("BMUnitID").to_dict()["Fuel type"]
        
        
        # datadescription ID of 143 = PN
        
        query_string = f"""
        
        SELECT TimeFrom, TimeTo, BMUnitID, LevelFrom, LevelTo 
        
        FROM PowerSystem.tblPhysicaldata as PD
        
        WHERE PD.TimeTo >= '{date_from}' AND PD.TimeTo <= '{date_to}' AND PD.DataDescriptionID = 143 AND PD.BMUnitID IN {Unit_IDs}
        
        """
        
        
        df = Data_Import().SQL_query().Custom_query(query_string = query_string)
        df["ft"] = df["BMUnitID"].map(ID_to_ft)
        df["BMU ID"] = df["BMUnitID"].map(ID_to_BMU)
        
        df.to_pickle(file_name)
        
    else:
        print("Reading pickle file")
        df = pd.read_pickle(file_name)
    return df
    

def Format_Data(df):

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
                df_new["ID"] = unit
                
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
                df_temp["ID"] = unit
                
                
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
    

    #df = pd.read_pickle("PN_train_data3.pickle")
    
    # =============================================================================
    # Below sorts out data
    # =============================================================================
    
    
    df = df.sort_values(by = "TimeTo")
    
    df = to_minute_granularity(df)
    
    
    fts = df["ft"].unique().tolist()
    IDs = df["ID"].unique().tolist()
    
    # This was coming from a coding error setting the entire first row to 0 if it was empty
    # rather than just the LevelTo column
    #raise ImportError("For some reason there's a random 0 in here?")
    
    ft_int_dict = {j: i for i, j in enumerate(fts)}
    int_ft_dict = {j: i for i, j in ft_int_dict.items()}
    
    ID_int_dict = {j: i for i, j in enumerate(IDs)}
    int_ID_dict = {j: i for i, j in ID_int_dict.items()}
    
    
    df["ft_int"] = df["ft"].map(ft_int_dict)
    #df["ID_int"] = df["ID"].map(ID_int_dict)
    
    
    df = df[df["ID"].isin(units)]
    
    df = df.sort_values(by = "TimeTo")
    df = df.set_index(["ft_int", "TimeTo"])[["LevelTo"]]
    
    return df, int_ft_dict


def Format_Test_Data(df):
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
        #print(len(ts))
        #print(max_time)
        #print(min_time)
        
        print(df)
        
        for i, unit in enumerate(units):
            #print(unit)
            dict_temp = df[df["BMU ID"] == unit].set_index(f"{time_col}").to_dict()[f"{target_col}"]
            
            """
            ft_temp = df[df["BMU ID"] == unit]["ft"].unique().tolist()
            #print(ft_temp)
            
            if len(ft_temp) > 1:
                raise ImportError(f"{unit} has multiple fuel types attributed to it. Please review and try again")
            else:
                ft_temp = ft_temp[0]
            """
            
            
            if i == 0:
                df_new = pd.DataFrame(index = ts)
                df_new[f"{target_col}"] = df_new.index.map(dict_temp)
                df_new["ID"] = unit
                
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
                df_temp["ID"] = unit
                
                
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
    
    
    df = df.sort_values(by = "TimeTo")
    df = to_minute_granularity(df)
    
    
    IDs = df["ID"].unique().tolist()

    
    ID_int_dict = {j: i for i, j in enumerate(IDs)}
    int_ID_dict = {j: i for i, j in ID_int_dict.items()}
    
    df["ID_int"] = df["ID"].map(ID_int_dict)
    
    return df

def nearest_divisible_int(value, divisor):
    
    # gets number of times divisor goes into value
    quotient = value // divisor
    
    # Find the two nearest numbers
    lower_nearest = quotient * divisor
    upper_nearest = (quotient + 1) * divisor
    
    # Determine which of the two is closer to the given value
    if abs(value - lower_nearest) < abs(value - upper_nearest):
        return lower_nearest
    else:
        return upper_nearest
    

units = ["E_DOLLB-1", "T_LARKS-1", "E_GOSHS-1"]
#units = ["E_DOLLB-1", "E_PILLB-1", "T_LARKS-1", "T_SUTBS-1", "E_GOSHS-1" "E_PETEM-3"]
#units = ["E_DOLLB-1", "T_SUTBS-1"]

PN_data = Load_PN_data(date_from = "2024-01-01", date_to = "2025-03-06", units = units, file_name = "TSC test data unit PNs.pickle")

fts = PN_data["ft"].unique().tolist()
df, int_ft_dict = Format_Data(PN_data)


if Model_Name not in os.listdir():
   
    
    # =============================================================================
    # Sets up training data
    # =============================================================================
    
    na_check = df[df.isna().any(axis = 1)]
    
    if len(na_check.index) > 0:
        print(na_check)
        raise ImportError("There are NA values in the dataset, please review and try again")
    

    
    # =============================================================================
    # Below ensures that the training dataset is of equal length, regardless of
    # how many unit types there are (you can't just do a flat x% of the main data set
    # as this can lead to one or more units having some of their data cut off, so they
    # dont have data for the same amount of seconds. This means the model won't run)
    # =============================================================================
    
    test_split = 0.8
    train_split = 0.2
    
    if test_split + train_split != 1:
        raise ImportError("test_split + train_split must equal 1")
    
    # checks if the test_split (defaulted to 80%) can be equally divided between the number of fuel types
    df_test_split = len(df.index)*test_split
    num_ft = len(fts)
    
    # if the train/test split isn't equally divisible by the number of technology types, the code finds the nearest number where this can be done 
    # and uses that as the train test split instead
    if df_test_split // num_ft != 0:
        print(df_test_split % num_ft)
        
        x = nearest_divisible_int(df_test_split, num_ft)
        
        df_train = df.iloc[:int(x)]
        df_test = df.iloc[int(x):]
    else:
        df_train = df.iloc[:int(df_test_split)]
        df_test = df.iloc[int(df_test_split):]
    
    
    X_train = df_train
    Y_train = pd.DataFrame(index = [i for i in range(len(fts))])
    Y_train["ft"] = Y_train.index.map(int_ft_dict)
    print(X_train)
    print(Y_train)
          
    print("Fitting ROCKET model...")
    clf = RocketClassifier(num_kernels=512) 
    clf.fit(X_train, Y_train) 
    
    print("Saving model")
    mlflow_sktime.save_model(sktime_model = clf, path = Model_Name)
    

else:
    print(f"Loading TSC model {Model_Name}...")
    clf = mlflow_sktime.load_model(model_uri = Model_Name) 

units = ["2__EFLEX002", "2__FANGE001", "T_SUTBS-1"]

#test_df = Data_Import().SQL_query().PN_data(date_from = "2024-01-01", date_to = "2025-01-01", BMU_IDs = units)
test_df = Data_Import().SQL_query().Physical_data(date_from = "2024-01-01", date_to = "2025-01-01", BMU_ID = "E_SOLUTIA")
test_df = Format_Test_Data(test_df)


test_df = test_df.sort_values(by = "TimeTo")
test_df = test_df.set_index(["ID", "TimeTo"])[["LevelTo"]]
print(test_df.value_counts())

print("Predicting...")
y_pred = clf.predict(test_df)
print(y_pred)

# Well the model runs but it is totally useless
# It identifies known peakers as batteries or solar sites (same for other tech types too)
# and it randomly fails sometimes saying that a value in the dataframe is infinity or too big for a float64 (which is total bollocks because I checked and no value like that exists)





