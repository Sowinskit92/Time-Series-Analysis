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
import time


class Data_Import:
    
    def __init__(self):
        
        # column renames for all of the SQL queries
        self.column_renames = {"SettlementDate": "Date", "HHPeriod": "SP", "DeliveryStartDate": "Start time", 
                               "DeliveryEndDate": "End time", "StartTime": "Start time", 
                               "DeliveryStart": "Start time", "ServiceDeliveryFromDate": "Start time",
                               "ServiceDeliveryToDate": "End time", "startTime": "Start time", "settlementDate": "Date",
                               "settlementPeriod": "SP", "Delivery Date": "Date", "eventStartTime": "Start time",
                               "eventEndTime": "End time",
                               
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
                               "elexonBmUnit": "BMU ID", "assetId": "BMU ID",
                               
                               "Unit_NGESOID": "NGU ID", "CompanyName": "Company", "NGESO_NGTUnitID": "NGU ID",
                               "leadPartyName": "Company",
                               
                               "GSPGroup": "GSP Group","ConstraintGroup": "Constraint",
                               
                               "ReportName": "Fuel type", "FuelTypeID": "Fuel type ID", "TechnologyType": "Fuel type",
                               "fuelType": "Fuel type",
                               
                               "BasketID": "Basket ID", "ServiceType": "Service type", "OrderType": "Order type", 
                               "AuctionProduct": "Service", "LoopedBasketID": "Looped Basket ID", 
                               
                               "CadlFlag": "CADL Flag", "SoFlag": "SO Flag", "StorFlag": "STOR Flag", "NivAdjustedVolume": "NIV volume",
                                
                               "demand": "Demand",
                               
                               "url": "URL", "id": "ID", 
                               
                               
                               
                               "normalCapacity": "Normal MW", "availableCapacity": "Available MW",
                               "unavailableCapacity": "Unavailable MW", "unavailabilityType": "Event type",
                               "eventStatus": "Status", "durationUncertainty": "Duration uncertainty",
                               "bmUnitName": "Name", "demandCapacity": "DC", "generationCapacity": "GC", 
                               "gspGroupId": "GSP Group"
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
                     API_from_joiner: str = "from", API_to_joiner: str = "to", fuel_types: list = None, 
                     REMIT: bool = False, only_last_revision: bool = False, profile_only: bool = False, 
                     REMIT_IDs = None):
            def Elexon_Data_Query(URL: str, start_date: str = None, end_date: str = None,
                           API_from_joiner: str = "from", API_to_joiner: str = "to", 
                           fuel_types: list = None, REMIT: bool = False, 
                           only_last_revision: bool = False, profile_only: bool = False,
                           REMIT_IDs: list = None):
                
                # =============================================================================
                # This function queries the Elexon API       
                # =============================================================================
                
                def Update_URL(URL: str, start_date: str = None, end_date: str = None,
                               API_from_joiner: str = "from", API_to_joiner: str = "to", 
                               fuel_types: list = None, REMIT: bool = False, 
                               only_last_revision: bool = True, profile_only: bool = False,
                               REMIT_IDs: list = None) -> str:
                    
                    # =============================================================================
                    # This function updates the URL to query the Elexon API, based on the 
                    # parameters entered
                    # =============================================================================
                    
                    if start_date != None:
                        URL = URL + f"&{API_from_joiner}={str(start_date)}"
                    if end_date != None:
                        URL = URL + f"&{API_to_joiner}={str(end_date)}"
                    if fuel_types != None:
                        for i in fuel_types:
                            URL = URL + f"&fuelType={i.upper()}"
                    if REMIT == True:
                        URL = URL + f"&lastestRevisionOnly={str(only_last_revision).lower()}"
                        URL = URL + f"&profileOnly={str(profile_only).lower()}"
                    if REMIT_IDs != None:
                        if isinstance(REMIT_IDs, list):
                            for i in REMIT_IDs:    
                                URL = URL + f"&messageId={i}"
                        else:
                            raise TypeError(f"REMIT_IDs need to be entered as a list. Please review and try again. Current format is: {REMIT_IDs}")
                    
                    
                    URL = URL + "&format=json"
                    
                    return URL
            
                
                URL_string = Update_URL(URL = URL, start_date = start_date, end_date = end_date,
                                        API_from_joiner = API_from_joiner, API_to_joiner = API_to_joiner, 
                                        fuel_types = fuel_types, REMIT = REMIT, 
                                        only_last_revision = only_last_revision,
                                        profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                
                try:
                    data = requests.get(URL_string).json()
                except OSError:
                    # if connection error raised, it will wait and re-try
                    sleep_time = 15
                    print(f"Error received - waiting {sleep_time}s to re-try")
                    time.sleep(sleep_time)
                    data = requests.get(URL_string).json()
                
                try:
                    df = pd.json_normalize(data, record_path = "data")
                except:
                    df = pd.json_normalize(data)
                
                return df
                
            
            if ID.startswith("/"):
                ID = ID[1:]
            else:
                pass
            
            URL_string = self.CONNECTION + ID + "?"
            
            if isinstance(REMIT_IDs, list):
                # deals with REMIT IDs
                # 450 = maximum amount of IDs Elexon lets you pull at a time
                MAX_REMIT_MESSAGES = 450
                if len(REMIT_IDs) <= MAX_REMIT_MESSAGES:
                    df = Elexon_Data_Query(URL = URL_string, REMIT = REMIT, 
                                                 only_last_revision = only_last_revision, 
                                                 profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                else:
                    # if len of IDs is greater than the max allowed values, it iterates through the list to collect them all
                    for i in range(int(len(REMIT_IDs)/MAX_REMIT_MESSAGES)):
                        diff_REMIT = len(REMIT_IDs) - ((i + 1)*MAX_REMIT_MESSAGES)
                        
                        n_start = i*MAX_REMIT_MESSAGES
                        n_end = (i + 1)*MAX_REMIT_MESSAGES
                        
                        print(f"{str(n_start)}: {str(n_end)}")
                        print(f"Remaining = {diff_REMIT}")
                        
                        if i == 0:
                            
                            df = Elexon_Data_Query(URL = URL_string, REMIT = REMIT, 
                                                         only_last_revision = only_last_revision, 
                                                         profile_only = profile_only,
                                                         REMIT_IDs = REMIT_IDs[n_start:n_end])
                        else:
                            
                            df_temp = Elexon_Data_Query(URL = URL_string, REMIT = REMIT, 
                                                         only_last_revision = only_last_revision, 
                                                         profile_only = profile_only,
                                                         REMIT_IDs = REMIT_IDs[n_start:n_end])
                            
                            df = pd.concat([df, df_temp])
                        

                    
                    if diff_REMIT == 0:
                        pass
                    else:
                        #diff_REMIT = len(REMIT_IDs) - ((i + 1)*MAX_REMIT_MESSAGES)
                        print(f"{str(n_end)}: {str(n_end + diff_REMIT)}")
                        
                        df_temp = Elexon_Data_Query(URL = URL_string, REMIT = REMIT, 
                                                     only_last_revision = only_last_revision, 
                                                     profile_only = profile_only,
                                                     REMIT_IDs = REMIT_IDs[n_end: n_end + diff_REMIT])
                        
                        df = pd.concat([df, df_temp])
            
            elif (start_date != None) and (end_date != None):
            
                start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days = 1)
                
                
                # calculates the difference in the requested period to gather data
                date_diff = (end_date_dt - start_date_dt).days
                # compares the time frame to the max Elexon's API allows for a given query
                if date_diff >= max_days:
                    # iterates through the time periods to gather all the data
                    
                    start_date_temp = start_date_dt.date()
                    
                    
                    while start_date_temp <= (end_date_dt - relativedelta(days = max_days)).date():
                        
                        
                        end_date_temp = (start_date_temp + relativedelta(days = max_days))
                        print(start_date_temp, end_date_temp)
                        """
                        URL_string_temp = Update_URL(URL = URL_string, start_date = start_date_temp,
                                                     end_date = end_date_temp, fuel_types = fuel_types, 
                                                     API_from_joiner = API_from_joiner,
                                                     API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                                     only_last_revision = only_last_revision, 
                                                     profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                        
                        
                        data = requests.get(URL_string_temp).json()
                        df_temp = pd.json_normalize(data, record_path = "data")
                        """
                        
                        df_temp = Elexon_Data_Query(URL = URL_string, start_date = start_date_temp,
                                                     end_date = end_date_temp, fuel_types = fuel_types, 
                                                     API_from_joiner = API_from_joiner,
                                                     API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                                     only_last_revision = only_last_revision, 
                                                     profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                        
                        try:
                            # concats data to df if it exists
                            df = pd.concat([df, df_temp])
                        except:
                            # otherwise creates dataframe
                            df = df_temp
                            
                        start_date_temp += relativedelta(days = max_days)
                    
                    # runs the final request
                    print(start_date_temp, end_date_dt.date())
                    """
                    URL_string_temp = Update_URL(URL = URL_string, start_date = start_date_temp,
                                                 end_date = end_date_dt.date(), fuel_types = fuel_types, 
                                                 API_from_joiner = API_from_joiner,
                                                 API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                                 only_last_revision = only_last_revision, 
                                                 profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                    
                    data = requests.get(URL_string_temp).json()
                    df_temp = pd.json_normalize(data, record_path = "data")
                    """
                    
                    df_temp = Elexon_Data_Query(URL = URL_string, start_date = start_date_temp,
                                                 end_date = end_date_dt.date(), fuel_types = fuel_types, 
                                                 API_from_joiner = API_from_joiner,
                                                 API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                                 only_last_revision = only_last_revision, 
                                                 profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                    
                    df = pd.concat([df, df_temp])
                    df = df.drop_duplicates(keep = "first")
                                    
                else:
                    """
                    URL_string = Update_URL(URL = URL_string, start_date = start_date,
                                            end_date = end_date, fuel_types = fuel_types, 
                                            API_from_joiner = API_from_joiner,
                                            API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                            only_last_revision = only_last_revision, 
                                            profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                    
                    data = requests.get(URL_string).json()
                    df = pd.json_normalize(data, record_path = "data")
                    """
                    
                    df = Elexon_Data_Query(URL = URL_string, start_date = start_date,
                                            end_date = end_date, fuel_types = fuel_types, 
                                            API_from_joiner = API_from_joiner,
                                            API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                            only_last_revision = only_last_revision, 
                                            profile_only = profile_only, REMIT_IDs = REMIT_IDs)
            
            else:
                # if there's no start_date or end_date
                """
                URL_string = Update_URL(URL = URL_string, fuel_types = fuel_types, 
                                        API_from_joiner = API_from_joiner, 
                                        API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                        only_last_revision = only_last_revision, 
                                        profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                
                data = requests.get(URL_string).json()
                df = pd.json_normalize(data, record_path = "data")
                """
                
                df = Elexon_Data_Query(URL = URL_string, fuel_types = fuel_types, 
                                        API_from_joiner = API_from_joiner, 
                                        API_to_joiner = API_to_joiner, REMIT = REMIT, 
                                        only_last_revision = only_last_revision, 
                                        profile_only = profile_only, REMIT_IDs = REMIT_IDs)
                
            df.rename(columns = Data_Import().column_renames, inplace = True)
            
            return df
          
        
        def BMU_data(self):
            df = Data_Import().Elexon_query().Get_Data("reference/bmunits/all")
            df = df[["BMU ID", "NGU ID", "Fuel type", "Company", "Name", 
                     "DC", "GC", "GSP Group"]]
            
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
            
        
        def REMIT(self, start_date: str, end_date: str, only_last_revision: bool = True,
                  profile_only: bool = False):
            
            REMIT_IDs = Data_Import().Elexon_query().Get_Data(ID = "remit/list/by-event", start_date = start_date,
                                                              end_date = end_date, REMIT = True, 
                                                              only_last_revision = only_last_revision,
                                                              profile_only = profile_only)
            #REMIT_IDs = REMIT_IDs.rename(columns = Data_Import().column_renames)
            
            REMIT_IDs = REMIT_IDs["ID"].unique().tolist()
            
            print(f"Number of REMIT IDs: {len(REMIT_IDs)}")
            
            REMIT = Data_Import().Elexon_query().Get_Data("/remit", REMIT_IDs = REMIT_IDs)
            REMIT = REMIT.rename(columns = Data_Import().column_renames)
            #print(REMIT.columns.tolist())
            REMIT = REMIT[["BMU ID", "Normal MW", "Available MW", 
                           "Unavailable MW", "Event type", "Status", 
                           "Start time", "End time", "Duration uncertainty"]]
            
            REMIT = REMIT.drop_duplicates(keep = "first")
            
            REMIT["Start time"] = pd.to_datetime(REMIT["Start time"])
            REMIT["End time"] = pd.to_datetime(REMIT["End time"])
            
            fuel_type_dict = pd.read_csv("BMU Info.csv").set_index("BMU ID").to_dict()["Fuel type"]
            REMIT["Fuel type"] = REMIT["BMU ID"].map(fuel_type_dict)
            REMIT["Fuel type"] = REMIT["Fuel type"].where(~REMIT["BMU ID"].str.startswith("I_"), "Interconnector")
            
            
            return REMIT
        
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