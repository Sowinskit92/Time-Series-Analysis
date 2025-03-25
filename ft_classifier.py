# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:35:21 2025

@author: Tim.Sowinski
"""

# system
import os
import sys

# data analysis
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from CI_data_imports22 import *

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

os.chdir("C:/Users/Tim.Sowinski/OneDrive - Cornwall Insight Ltd/Documents/Flex Stuff/Flex_Digitalisation/TSC testing/XGBoost (attempt 2)")

# =============================================================================
# This is the main file for training the model and testing it 
# (ie it combines the other two files I've already made)
# =============================================================================



def load_KnownFuelTypes(fuel_types: list) -> dict:
    
    # =============================================================================
    # Returns a dict of BMUs of the specified fuel types that aren't additional units
    # =============================================================================
    
    BMUs = Data_Import().SQL_query().BMU_data()
    BMUs = BMUs[(BMUs["Fuel type"].isin(fuel_types)) & (BMUs["BMU ID"].str[:2].isin(["T_", "E_"]))]
    
    BMUs = BMUs.sort_values(by = "BMU ID")
    BMUs = BMUs.set_index("BMU ID").to_dict()["Fuel type"]
    return BMUs
    

def load_UnknownFuelTypes() -> list:
    # =============================================================================
    # Returns a list of additional/secondary BMUs 
    # =============================================================================
    
    BMUs = Data_Import().SQL_query().BMU_data()
    print(BMUs["Fuel type"].unique().tolist())
    
    BMUs = BMUs[(~BMUs["Fuel type"].isin(["Wind", "Demand", "Biomass", "NPSHYD", "Synchronous Condenser"])) & (BMUs["BMU ID"].str[:2].isin(["2_", "V_"])) & (~BMUs["BMU ID"].str.endswith("000"))]
    
    
    BMUs = sorted(BMUs["BMU ID"].unique().tolist())
    return BMUs


def gather_Data(units: dict, date_from: str, date_to = str((datetime.now() - relativedelta(days = 7)).date()), file_name = "PNs.pickle"):
    
    
    if file_name not in os.listdir():
        
        if isinstance(units, dict):
            units_list = list(units.keys())
            units = list(units.keys())
        elif isinstance(units, list):
            units_list = units
        else:
            raise TypeError("units must be a dictionary with BMU IDs as the keys and fuel types as the values")
        
        units_list = f"{units_list}"
        units_list = units_list.replace("[", "(")
        units_list = units_list.replace("]", ")")
        
        # query for PN data
        query = f"""
        
        SELECT TimeTo, LevelTo, Elexon_BMUnitID
        
        FROM PowerSystem.tblPhysicalData as PD
        
        INNER JOIN META.tblBMUnit_Managed as BMU on PD.BMUnitID = BMU.BMUnitID
        WHERE Elexon_BMUnitID in {units_list} AND DataDescriptionID = 143 AND TimeTo >= '{date_from}' AND TimeTo <= '{date_to}'
        
        
        """
        print(date_from, date_to)
        # =============================================================================
        # Creates test dataframe down to minute granularity
        # =============================================================================
        date_to = datetime.strptime(date_to, "%Y-%m-%d") + relativedelta(days = 1)
        dr = pd.date_range(date_from, str(date_to.date()), freq = "1min")
        for i, j in enumerate(units):
            if i == 0:
                df = pd.DataFrame(index = dr)
                df["id"] = j
            else:
                df_temp = pd.DataFrame(index = dr)
                df_temp["id"] = j
                df = pd.concat([df, df_temp])
        
        df = df.reset_index().rename(columns = {"index": "time"})
        #print(df)
        PNs = Data_Import().SQL_query().Custom_query(query)
        
        
        df = pd.merge(df, PNs, left_on = ["time", "id"], right_on = ["TimeTo", "BMU ID"], how = "outer")
        del df["TimeTo"], df["BMU ID"]
        df["LevelTo"] = df["LevelTo"].ffill()
        print(df)
        df.to_pickle(file_name)
    else:
        df = pd.read_pickle(file_name)
    
    return df


def format_Data(df, integrated: bool == False):
    
    # returns data in minute granularity
    if integrated == False:
        df["mod"] = "min_" + ((df["time"].dt.hour)*60 + df["time"].dt.minute).astype(str) # minute of day
        df["date"] = df["time"].dt.date
        

        FEATURES = df["mod"].unique().tolist()
        TARGET = "ft"

        print("Pivoting data...")
        df = pd.pivot_table(df, index = ["date", "id"], columns = "mod", values = "LevelTo").reset_index()
        
        # removes final day of data as these will likely be null values
        df = df[df["date"] < df["date"].max()]
        
        
        df["sum"] = df[[i for i in df.columns.tolist() if "min_" in i]].abs().sum(axis = 1)
        df["max"] = df[[i for i in df.columns.tolist() if "min_" in i]].abs().max(axis = 1)
        
        
        # removes entries which have only 0 values for FPNs as this could cause issues with model training
        df = df[df["sum"] != 0]
        
        
        # reframes PN data so that it's as a fraction of the max absolute FPN for the day
        for i in [i for i in df.columns.tolist() if "min_" in i]:
            df[i] = df[i].div(df["max"])
            
        del df["sum"], df["max"]
    
    else: # returns data integrated over each 30min period
        df["period"] = df["time"].dt.floor("30T").dt.time
        # turns into settlement periods
        SPs = {period: f"SP_{idx + 1}" for idx, period in enumerate(df["period"].unique().tolist())}
        df["period"] = df["period"].map(SPs)
        df["date"] = df["time"].dt.date
        
        FEATURES = list(SPs.values())
        TARGET = "ft"
        
        print("Pivoting data...")
        df = pd.pivot_table(df, index = ["date", "id"], columns = "period", values = "LevelTo", aggfunc = "sum").reset_index()
        
        # removes final day of data as these will likely be null values
        df = df[df["date"] < df["date"].max()]
        
        
        df["sum"] = df[[i for i in df.columns.tolist() if "SP_" in i]].abs().sum(axis = 1)
        df["max"] = df[[i for i in df.columns.tolist() if "SP_" in i]].abs().max(axis = 1)
        
        
        # removes entries which have only 0 values for FPNs as this could cause issues with model training
        df = df[df["sum"] != 0]
        
        
        # reframes PN data so that it's as a fraction of the max absolute energy in an SP for the day
        for i in [i for i in df.columns.tolist() if "SP_" in i]:
            df[i] = df[i].div(df["max"])
        
        
        del df["sum"], df["max"]
        
    
    return df, FEATURES, TARGET
    
    

def train_Model(training_units: dict, date_from: str = None, min_training_date: str = str((datetime.now() - relativedelta(years = 2)).date()), 
                model_Name = "Fuel_type_classifier.json", pivoted_data_filename = "pivoted train data.pickle", integrated: bool = False):
    
    # =============================================================================
    # ---------------- PARAMETERS ----------------
    # trading_units ----> dictionary of units to train model on (key = BMU ID, value = fuel type)
    # date_from     ----> earliest date to gather physical data from. If blank, defaults to min_training_date
    # min_training_date ---> minimum range of physical data wanted to train the model
    # =============================================================================
    
    # =============================================================================
    # Functions
    # =============================================================================
    
    def find_MinDate(units: list):
        print("Finding min date for training data...")
        # =============================================================================
        # Searches unit list and retrieves the earliest date we have physical data
        # for them. Then returns the max of these, so we can gather data from then
        # =============================================================================
        units = f"{units}"
        units = units.replace("[", "(")
        units = units.replace("]", ")")
        query = f"""
        SELECT MIN(TimeTo) as date, Elexon_BMUnitID
        FROM PowerSystem.tblPhysicalData as PD
        INNER JOIN META.tblBMUnit_Managed as BMU on PD.BMUnitID = BMU.BMUnitID
        WHERE Elexon_BMUnitID in {units}
        GROUP BY Elexon_BMUnitID
        """
        file_name = "BMU min dates.pickle"
        if file_name not in os.listdir():
            earliest_dates = Data_Import().SQL_query().Custom_query(query)
            print(earliest_dates)
            earliest_dates["date"] = pd.to_datetime(earliest_dates["date"])
            date = earliest_dates["date"].max()
            print(date)
            
            earliest_dates.to_pickle(file_name)
        else:
            # doesn't take into account if the units change
            earliest_dates = pd.read_pickle(file_name)
            date = earliest_dates["date"].max()
        
        return earliest_dates
    
    
    # =============================================================================
    # Code
    # =============================================================================
    
    BMU_earliest_dates = find_MinDate(list(training_units.keys()))
        
    # removes units where their minimum date less than min training date
    BMU_earliest_dates = BMU_earliest_dates[BMU_earliest_dates["date"] <= datetime.strptime(min_training_date, "%Y-%m-%d")]
    
    if date_from == None:
        # gets the maximum earliest date 
        date_from = BMU_earliest_dates["date"].max().date()
    
    training_units = {unit: ft for unit, ft in training_units.items() if unit in BMU_earliest_dates["BMU ID"].unique().tolist()}
    print("\nTraining dataset overview:")
    training_info = pd.DataFrame(index = range(len(training_units)), data = {"BMU ID": list(training_units.keys()), "ft": list(training_units.values())})
    print(training_info["ft"].value_counts())
    print()

    ft_to_id = {ft: idx for idx, ft in enumerate(list(set(training_units.values())))}
    ft_ids = pd.DataFrame(data = {"id": [i for i in ft_to_id.values()], "Fuel type": [i for i in ft_to_id.keys()]})
    ft_ids.to_csv("fuel_type_ids.csv", index = False)
    
    print(ft_to_id)
    if pivoted_data_filename not in os.listdir():
        print("Gathering data...")
        df = gather_Data(units = training_units, date_from = date_from, file_name = "Train PNs.pickle")
        print(df)
        sys.exit()
        # just for testing
        #df = df.iloc[:1_000_000]
        df, FEATURES, TARGET = format_Data(df, integrated = integrated)
        df.to_pickle(pivoted_data_filename)
    else:
        df = pd.read_pickle(pivoted_data_filename)
        FEATURES = [i for i in df.columns.tolist() if ("min_" in i) or ("SP_" in i)]
        TARGET = "ft"
    
    df["ft"] = df["id"].map(training_units)
    # =============================================================================
    # sets up training and testing data
    # =============================================================================
    # converts fuel type to numeric values
    df[TARGET] = df[TARGET].map(ft_to_id)
    X = df[FEATURES] # features are either the SPs or the minute of day
    Y = df[TARGET]
    
    
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 22)
    
    # =============================================================================
    # Sets up model
    # =============================================================================
    print("Fitting model...")
    
    params = {"objective": "reg:logistic", 
              "max_depth": 5,
              "alpha": 1, 
              "learning_rate": 0.1,
              "n_estimators": 90}
    
    model = XGBClassifier(**params)
    model.fit(X_train, Y_train)
    print(f"Model score: {model.score(x_test, y_test)}")
    
    print("Predicting...")
    y_pred = model.predict(x_test)
    CM = confusion_matrix(y_test, y_pred, labels = list(ft_to_id.values()))
    
    print("Confusion matrix:")
    print(CM)
    # =============================================================================
    # Saving model
    # =============================================================================
    print(f"Saving model as {model_Name}...")
    model.save_model(model_Name)
    
    return model, ft_to_id
    
    
def classify_FuelTypes(date_from: str, date_to: str, units: list = None, file_name = "unknown_BMU_PNs.pickle", pivoted_file_name = "unknown_BMU_PNs_pivoted.pickle", model_Name = "Fuel_type_classifier.json"):
    
    print("Loading model...")
    model = XGBClassifier()
    model.load_model(model_Name)
    
    
    
    if units == None:
        units = load_UnknownFuelTypes()
    print(f"{len(units)} units identified to classify over range of: {date_from} - {date_to}")
    
    #units = units[:100] # just for testing
    if pivoted_file_name not in os.listdir():
        print("Gathering PNs of units to be classified...")
        if file_name not in os.listdir():
            PNs = gather_Data(units = units, date_from = date_from, date_to = date_to, file_name = file_name)
            PNs.to_pickle(file_name)
        else:
            PNs = pd.read_pickle(file_name)
        
        #print("\nPNs:")
        PNs, FEATURES, TARGET = format_Data(PNs, integrated = False)
        #print("\nPivoted PNs:")
        #print(PNs)
        PNs.to_pickle(pivoted_file_name)
    else:
        PNs = pd.read_pickle(pivoted_file_name)
        FEATURES = [i for i in PNs.columns.tolist() if ("SP_" in i) or ("min_" in i)]
        print("\nPivoted PNs:")
    
    dates = PNs["date"].unique().tolist()
    
    fuel_type_ids = pd.read_csv("fuel_type_ids.csv")
    fuel_type_ids = fuel_type_ids.set_index("id").to_dict()["Fuel type"]
    
    ft_df = pd.DataFrame()
    
    for idx, i in enumerate(dates):
        print(i)
        df = PNs[PNs["date"] == i]
        X = df[FEATURES]
        #print(X)
        pred = model.predict(X)
        
        df["ft"] = pred
        df["ft"] = df["ft"].map(fuel_type_ids)
        #print(df)
        
        if idx == 0:
            ft_df = df[["date", "id", "ft"]]
        else:
            ft_df = pd.concat([ft_df, df[["date", "id", "ft"]]])
    
    
    print(ft_df)
    ft_df = ft_df.set_index(["date", "id"])
    ft_df.to_excel(f"{model_Name} - classified fuel types.xlsx")
    
    return ft_df
    print("The data is still picking up units on days where there's no data available for them (2__AOXPO001, 2__APOWQ001 on 1 Jan 2025). Also 2__AMRCY001 may not be a battery but it classifies it as one")
    print("Could be good to run this first, then let anything else override the results")
    print("Also using other physical data would probably help quite a lot, as PNs can be zero but the MILs/MELs are changing in response to its activity")
    sys.exit()
    
    
# other option could be to integrate the volume over each SP. Integrated model does not work as well

model_Name = "Fuel_type_classifier.json"

if model_Name not in os.listdir():
    # gathers all non-additional units on server with a fuel type in the specified list
    training_units = load_KnownFuelTypes(["Solar", "Gas Recip", "Battery"])
    train_Model(model_Name = model_Name, training_units = training_units, min_training_date = "2023-10-01", pivoted_data_filename = "pivoted train data.pickle", integrated = False)
else:
    df = classify_FuelTypes(date_from = "2025-01-01", date_to = "2025-01-10")
    
    
        
        







