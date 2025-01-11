#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:11:25 2024

@author: timsowinski
"""

import pandas as pd
from datetime import datetime, timedelta
import requests
import pyodbc
import json
import sys
import os
from dateutil.relativedelta import relativedelta
import warnings
import xlwings as xw


class Data_Import:
    
    def __init__(self):
        
        # column renames for all of the SQL queries
        self.column_renames = {"SettlementDate": "Date", "HHPeriod": "SP", "DeliveryStartDate": "Start time", 
                               "DeliveryEndDate": "End time", "StartTime": "Start time", 
                               "DeliveryStart": "Start time", "ServiceDeliveryFromDate": "Start time",
                               "ServiceDeliveryToDate": "End time", "startTime": "Start time", "settlementDate": "Date",
                               "settlementPeriod": "SP", "Delivery Date": "Date",
                               
                               "TimeFrom": "Time from",
                               
                               "Value": "Price", "PriceLimit": "Submitted price", 
                               "ClearingPrice": "Clearing price", "FinalPrice": "Final price", 
                               "Utilisation Price GBP per MWh": "Utilisation price (£/MWh)",
                               "TenderedAvailabilityPrice": "Availability price", 
                               "MarketClearingPrice": "Clearing price", "price": "Price",
                               
                               "Volume": "Submitted volume", "Volume(MW)": "Submitted volume",
                               "ExecutedVolume": "Accepted volume", "AcceptedVolume(MW)": "Accepted volume",
                               "TenderedMW": "Submitted volume", "ContractedMW": "Accepted volume",
                               "volume": "Volume", "generation": "MW",
                               
                               "Elexon_BMUnitID": "BMU ID", "NGC_BMUnitID": "NGU ID", "PartyName": "Company", 
                               "Registered DFS Participant": "Company", "nationalGridBmUnit": "NGU ID",
                               "elexonBmUnit": "BMU ID",
                               
                               "Unit_NGESOID": "NGU ID", "CompanyName": "Company", "NGESO_NGTUnitID": "NGU ID",
                               
                               "GSPGroup": "GSP Group","ConstraintGroup": "Constraint",
                               
                               "ReportName": "Fuel type", "FuelTypeID": "Fuel type ID", "TechnologyType": "Fuel type",
                               "fuelType": "Fuel type",
                               
                               "BasketID": "Basket ID", "ServiceType": "Service type", "OrderType": "Order type", 
                               "AuctionProduct": "Service", "LoopedBasketID": "Looped Basket ID", 
                               
                               "CadlFlag": "CADL Flag", "SoFlag": "SO Flag", "StorFlag": "STOR Flag", "NivAdjustedVolume": "NIV volume",
                                
                               "demand": "Demand"
                               }
    
    
        
    class NESO_query():
        
        def __init__(self, date_from: str = False, date_to: str = False):
            self.date_from = date_from
            self.date_to = date_to
            self.CONNECTION = "https://api.neso.energy/api/3/action/datastore_search?resource_id="
        
        def Get_Data(self, ID, n_results: int = 10000):
            # returns data from NESO's API URL string for NESO API
            # ID for dataset can be found from NESO's data portal
            URL_string = self.CONNECTION + ID + f"&limit={str(n_results)}"
            # print(URL_string)
            data = requests.get(URL_string).json()
            df = pd.json_normalize(data, record_path = ["result", "records"])
            df.rename(columns = Data_Import().column_renames, inplace = True)
            del df["_id"]
            return df
            
        def BR_Requirements(self):
            df = Data_Import().NESO_query().Get_Data("a9470515-c1de-4c4b-9597-2d78bf24c29f")
            return df
        
        def DFS_Utilisation_data(self):
            print("Gathering DFS Utilisation data")
            df = Data_Import().NESO_query().Get_Data("25698259-0b66-42f0-ac59-ef0df5245812")
            return df
        
        def DFS_Submissions_data(self):
            print("Gathering DFS Submission data")
            df = Data_Import().NESO_query().Get_Data("cc36fff5-5f6f-4fde-8932-c935d982ecd8")
            return df
        
        def Operating_Plan_data(self):
            print("Gathering Operating Plan data")
            df = Data_Import().NESO_query().Get_Data("e51f2721-00ab-4182-9cae-3c973e854aa8")
            return df
        
        def QR_Requirements_data(self):
            print("Gathering QR Requirement data")
            df = Data_Import().NESO_query().Get_Data("f012de08-b258-408b-bc41-f885e183f97f")
            return df
    
    class Elexon_query():
        def __init__(self, date_from: str = False, date_to: str = False):
            self.date_from = date_from
            self.date_to = date_to
            self.CONNECTION = "https://data.elexon.co.uk/bmrs/api/v1/"
        
        
        
        def Get_Data(self, ID, start_date: str = None, end_date: str = None, max_days = 7,
                     API_from_joiner: str = "from", API_to_joiner: str = "to", fuel_types: list = None):
            
            if ID.startswith("/"):
                ID = ID[1:]
            else:
                pass
            
            URL_string = self.CONNECTION + ID + "?"
            
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days = 1)
            
            
            # calculates the difference in the requested period to gather data
            date_diff = (end_date_dt - start_date_dt).days
            # compares the time frame to the max Elexon's API allows for a given query
            if date_diff >= max_days:
                # iterates through the time periods to gather all the data
                
                start_date_temp = start_date_dt.date()
                
                
                while start_date_temp <= (end_date_dt - relativedelta(days = max_days)).date():
                    
                    URL_string_temp = URL_string
                    
                    end_date_temp = (start_date_temp + relativedelta(days = max_days))
                    print(start_date_temp, end_date_temp)
                    
                    if start_date != None:
                        URL_string_temp = URL_string_temp + f"&{API_from_joiner}={str(start_date_temp)}"
                    if end_date != None:
                        URL_string_temp = URL_string_temp + f"&{API_to_joiner}={str(end_date_temp)}"
                    if fuel_types != None:
                        for i in fuel_types:
                            URL_string_temp = URL_string_temp + f"&fuelType={i.upper()}"
                    
                    URL_string_temp = URL_string_temp + "&format=json"
                    
                    data = requests.get(URL_string_temp).json()
                    df_temp = pd.json_normalize(data, record_path = "data")
                    
                    try:
                        # concats data to df if it exists
                        df = pd.concat([df, df_temp])
                    except:
                        # otherwise creates dataframe
                        df = df_temp
                        
                    start_date_temp += relativedelta(days = max_days)
                
                # runs the final request
                print(start_date_temp, end_date_dt.date())
                URL_string_temp = URL_string + f"&{API_from_joiner}={str(start_date_temp)}&{API_to_joiner}={str(end_date_dt.date())}"
                if fuel_types != None:
                    for i in fuel_types:
                        URL_string_temp = URL_string_temp + f"&fuelType={i.upper()}"
                
                URL_string_temp = URL_string_temp + "&format=json"
                
                data = requests.get(URL_string_temp).json()
                df_temp = pd.json_normalize(data, record_path = "data")
                df = pd.concat([df, df_temp])
                df = df.drop_duplicates(keep = "first")
                                
            else:
                if start_date != None:
                    URL_string = URL_string + f"&{API_from_joiner}={start_date}"
                if end_date != None:
                    URL_string = URL_string + f"&{API_to_joiner}={end_date}"
                if fuel_types != None:
                    for i in fuel_types:
                        URL_string = URL_string + f"&fuelType={i.upper()}"
                
                URL_string = URL_string + "&format=json"
                data = requests.get(URL_string).json()
                df = pd.json_normalize(data, record_path = "data")
                
            df.rename(columns = Data_Import().column_renames, inplace = True)
            
            return df
            
        def BMU_data(self):
            df = Data_Import().Elexon_query().Get_Data("reference/bmunits/all")
            return df
        
        def MIP_data(self, start_date: str, end_date: str):
            
            
            df = Data_Import().Elexon_query().Get_Data("/balancing/pricing/market-index",
                                                       start_date = start_date, end_date = end_date)
            df["Start time"] = pd.to_datetime(df["Start time"])
            
            df = df[df["dataProvider"] == "APXMIDP"]
            
            df = df.sort_values(by = "Start time").reset_index(drop = True)
            # last row is always the beginning of the next day, so this removes it
            df = df.iloc[:len(df.index) - 1]
            
            return df
        
        def Demand_data(self, start_date: str, end_date: str):
            
            df = Data_Import().Elexon_query().Get_Data(ID = "datasets/ITSDO", start_date = start_date, end_date = end_date,
                                                        API_from_joiner = "publishDateTimeFrom", 
                                                        API_to_joiner = "publishDateTimeTo")
            df = df.rename(columns = Data_Import().column_renames)
            df["Start time"] = pd.to_datetime(df["Start time"])
            df = df.sort_values(by = "Start time").reset_index(drop = True)
            del df["publishTime"]
            
            return df
            
        def Fuel_Mix_data(self, start_date: str, end_date: str, fuel_types: list = None):
            
            df = Data_Import().Elexon_query().Get_Data(ID = "datasets/FUELHH", start_date = start_date, end_date = end_date,
                                                        API_from_joiner = "publishDateTimeFrom", 
                                                        API_to_joiner = "publishDateTimeTo", fuel_types = fuel_types)
            
            df = df.rename(columns = Data_Import().column_renames)
            df["Start time"] = pd.to_datetime(df["Start time"])
            
            df = df.sort_values(by = "Start time").reset_index(drop = True)
            del df["publishTime"], df["dataset"]
            
            
            return df
            
    
    class Help():
        # can call this to show the different data available in the different classes
        def __init__(self):
            print()

def Time_Add(df, col_to_transform = "SP", As = "UTC", time_col_name = "Time"):
    # takes a column and transforms it to a time value
    
    
    if (col_to_transform == "SP") and (As == "UTC"):
        df["mins"] = (df[col_to_transform] - 1).mul(30)
        df[time_col_name] = pd.to_datetime(df["mins"], unit = "m").dt.strftime("%H:%M")
        del df["mins"]
        return df
    else:
        raise AttributeError("Time column can't be added if not SP and in UTC currently")

def Excel_load(sheet_name, data, cell_ref, name = False, clear_range = False): #if using coordinates, needs to be entered as a list like [(row1, col1), (row2, col2)]
    """For cell_ref coordinates, must be in the format [(row, col)]. For clear range coordinates, must be in the format [(row1, col1), (row2, col2)]"""     
    current_sheets = []
    for i in range(len(workbook.sheets)):
        current_sheets.append(workbook.sheets[i].name) #this returns a new list of just the sheet names
    
    #adds in sheets if it doesn't already exist
    if sheet_name not in current_sheets:
        workbook.sheets.add("{}".format(sheet_name))
        current_sheets.append("{}".format(sheet_name))
    else:
        pass
    
    sheet = workbook.sheets["{}".format(sheet_name)]
    
    if name == False:
        name = " "
    else:
        name = name
    
    if clear_range != False:
        
        if type(clear_range) == list and type(cell_ref) == list:
            #print("A")
            sheet.range(clear_range[0], clear_range[1]).clear_contents()
            #print("Cleared")
            sheet[cell_ref[0]].value = data
            sheet[cell_ref[0]].value = name
        elif type(clear_range) == list and type(cell_ref) == str:
            #print("B")
            sheet.range(clear_range[0], clear_range[1]).clear_contents()
            sheet["{}".format(cell_ref)].value = data
            sheet["{}".format(cell_ref)].value = name
        elif type(clear_range) == str and type(cell_ref) == list:
            #print("C")
            sheet.range("{}".format(clear_range)).clear_contents()
            sheet[cell_ref[0]].value = data
            sheet[cell_ref[0]].value = name
        elif isinstance(clear_range, str) and isinstance(cell_ref, tuple):
            #print("D")
            sheet.range("{}".format(clear_range)).clear_contents()
            sheet[cell_ref].value = data
            sheet[cell_ref].value = name
        elif isinstance(clear_range, list) and isinstance(cell_ref, tuple):
            #print("E")
            sheet.range(clear_range[0], clear_range[1]).clear_contents()
            sheet[cell_ref].value = data
            sheet[cell_ref].value = name
        else:
            #print("F")
            sheet.range(clear_range).clear_contents()
            sheet[cell_ref].value = data
            sheet[cell_ref].value = name
        
    else:
        if type(cell_ref) == list:
            sheet[cell_ref[0]].value = data
            sheet[cell_ref[0]].value = name
        elif isinstance(cell_ref, tuple):
            sheet[cell_ref].value = data
            sheet[cell_ref].value = name
        elif type(cell_ref) == str:
            sheet["{}".format(cell_ref)].value = data
            sheet["{}".format(cell_ref)].value = name
        else:
            print("Loading cell reference for {} not in the required format of [(row1, col1)] or string".format(sheet_name))


if __name__ == "__main__":
    pass
    print(f"Data collected in {datetime.now() - start_time}")
else:
    pass