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


pd.set_option("display.max_columns", None)


"""
The mrID one works well but there's too many mrIDs for it to be feasible, Elexon just stops you
from querying it

So I'm thinking about going back to the ID list but making sure the onlylastrevision argument is True

But I'm going to overhaul the REMIT function as that's confusing now

"""


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
                               "leadPartyName": "Company", "affectedUnit": "NGU ID",
                               
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
                               "gspGroupId": "GSP Group", 
                               
                               "revisionNumber": "Revision", "createdTime": "Created time", 
                               "publishTime": "Published time"
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
        
        
        def Get_Data(self, ID, start_date: str = None, end_date: str = None,
                     API_from_joiner: str = "from", API_to_joiner: str = "to", 
                     fuel_types: list = None, REMIT: bool = False, 
                     only_last_revision: bool = False, profile_only: bool = False,
                     REMIT_IDs: list = None, TEST: bool = False):
            
            def Elexon_Data_Query(URL: str, start_date: str = None, end_date: str = None,
                                  API_from_joiner: str = "from", API_to_joiner: str = "to", 
                                  fuel_types: list = None, REMIT: bool = False, 
                                  only_last_revision: bool = False, profile_only: bool = False,
                                  REMIT_IDs: list = None, TEST: bool = False):
                
                # =============================================================================
                # This function queries the Elexon API       
                # =============================================================================
                
                def Update_URL(URL: str, start_date: str = None, end_date: str = None,
                               API_from_joiner: str = "from", API_to_joiner: str = "to", 
                               fuel_types: list = None, REMIT: bool = False, 
                               only_last_revision: bool = True, profile_only: bool = False,
                               REMIT_IDs: list = None, TEST: bool = False) -> str:
                    
                    # =============================================================================
                    # This function updates the URL to query the Elexon API, based on the 
                    # parameters entered
                    # =============================================================================
                    
                    # removes the / if accidentally put in
                    if URL.startswith("/"):
                        URL = URL[1:]
                        
                    URL = self.CONNECTION + URL + "?"
                    
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
                            # max len Elexon allows is ~480
                            if len(REMIT_IDs) <= 480:
                                for i in REMIT_IDs[:480]:
                                    URL = URL + f"&messageId={i}"
                            else:
                                raise AttributeError(f"Length of REMIT_IDs list cannot exceed 480. Current length is {len(REMIT_IDs)}")
                            
                            

                        else:
                            raise AttributeError("Please enter the REMIT IDs as a list")
                    
                    URL = URL + "&format=json"
                    #print(URL)
                    
                    if TEST == True:
                        print(URL)
                    else:
                        pass
                    
                    # =============================================================================
                    # End of Update_URL
                    # =============================================================================
                    
                    return URL
                
                if REMIT_IDs == None:
                    URL_string = Update_URL(URL = URL, start_date = start_date, end_date = end_date,
                                            API_from_joiner = API_from_joiner, API_to_joiner = API_to_joiner, 
                                            fuel_types = fuel_types, REMIT = REMIT, 
                                            only_last_revision = only_last_revision,
                                            profile_only = profile_only, REMIT_IDs = REMIT_IDs, 
                                            TEST = TEST)
                    data = requests.get(URL_string).json()
                    
                    try:
                        df = pd.json_normalize(data, record_path = "data")
                    except:
                        df = pd.json_normalize(data)
                else:
                    # deals with gathering REMIT IDs
                    # number of iterations to get all the REMIT IDs
                    itr = len(REMIT_IDs)/480
                    for i in range(int(itr) + 1):
                        
                        if i < int(itr):
                            # normal iterations 
                            #print("YEAH")
                            n_i = 480*i
                            n_f = 480*(i+1) - 1
                            print(f"Gathering REMIT IDs {n_i} to {n_f} out of {len(REMIT_IDs)}...")
                        else:
                            #print("UEHFBN")
                            # final iteration
                            n_i = 480*i
                            n_f = len(REMIT_IDs)
                            print(f"Gathering REMIT IDs {n_i} to {n_f} out of {len(REMIT_IDs)}...")
                        
                        
                        
                        RIDs_temp = REMIT_IDs[n_i: n_f]
                        
                        URL_string = Update_URL(URL = URL, start_date = start_date, end_date = end_date,
                                                API_from_joiner = API_from_joiner, API_to_joiner = API_to_joiner, 
                                                fuel_types = fuel_types, REMIT = REMIT, 
                                                only_last_revision = only_last_revision,
                                                profile_only = profile_only, REMIT_IDs = RIDs_temp, 
                                                TEST = TEST)
                        
                        if i == 0:
                            data = requests.get(URL_string).json()
                            try:
                                df = pd.json_normalize(data, record_path = "data")
                            except:
                                df = pd.json_normalize(data)
                        else:
                            data_temp = requests.get(URL_string).json()
                            try:
                                df_temp = pd.json_normalize(data_temp, record_path = "data")
                            except:
                                df_temp = pd.json_normalize(data_temp)
                            
                            df = pd.concat([df, df_temp]).reset_index(drop = True)
                    
                            
                
                
                if TEST == True:
                    print(df)
                    sys.exit()
                else:
                    pass
                
                
                # =============================================================================
                # End of Elexon Data Query            
                # =============================================================================
                
                return df
            
            df = Elexon_Data_Query(URL = ID, start_date = start_date, end_date = end_date, 
                                   API_from_joiner = API_from_joiner, API_to_joiner = API_to_joiner, 
                                   fuel_types = fuel_types, REMIT = REMIT, only_last_revision = only_last_revision,
                                   profile_only = profile_only, REMIT_IDs = REMIT_IDs, TEST = TEST)
            
        
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
            
        
        def REMIT(self, start_date: str, end_date: str, only_last_revision: bool = True):
            
            IDs = Data_Import().Elexon_query().Get_Data(ID = "remit/list/by-event/stream", 
                                                       start_date = start_date, 
                                                       end_date = end_date, REMIT = True, 
                                                       only_last_revision = True)
            IDs = IDs["id"].unique().tolist()
            REMIT = Data_Import().Elexon_query().Get_Data(ID = "/remit", REMIT_IDs = IDs, 
                                                          TEST = False)
            REMIT = REMIT.rename(columns = Data_Import().column_renames)
            REMIT = REMIT[["Created time", "Start time", "End time", "Event type", "BMU ID", 
                           "Fuel type", "Normal MW", "Available MW", "Unavailable MW", 
                           "Status"]]
            REMIT["Start time"] = pd.to_datetime(REMIT["Start time"])
            REMIT["End time"] = pd.to_datetime(REMIT["End time"])
            
            REMIT = REMIT.drop_duplicates(keep = "first")
            
            fuel_type_dict = pd.read_csv("BMU Info.csv").set_index("BMU ID").to_dict()["Fuel type"]
            REMIT["Fuel type"] = REMIT["BMU ID"].map(fuel_type_dict)
            REMIT["Fuel type"] = REMIT["Fuel type"].where(~REMIT["BMU ID"].str.startswith("I_"), "Interconnector")
            
            
            # =============================================================================
            # Creates SP start and end times     
            # =============================================================================
            
            REMIT["SP_s"] = REMIT["Start time"].dt.round("30min").dt.tz_localize(None)
            REMIT["SP_e"] = REMIT["End time"].dt.round("30min").dt.tz_localize(None)
            print(REMIT)
            
            
            
            """Now need to try sum the unavailable capacity between the periods here"""
            
            dr_start = pd.date_range(start = start_date, end = end_date, freq = "30min")
            dr_end = dr_start + pd.DateOffset(days = 1/48)
            
            
            df = pd.DataFrame(index = dr_start, data = dr_end).reset_index()
            df.rename(columns = {"index": "Start time", 0: "End time"}, inplace = True)
            
            """Groupby REMIT by start date, then do the same for end date, then you should be 
            able to just minus them from each other no? (ie amount coming online - amount going offline
                                                         should equal the total amount online?)"""
            
            
            
            for i in sorted(REMIT["Fuel type"].astype(str).unique().tolist()):
                
                # having checked this method on 19 Jan 2025, I'm happy with the results it's putting out
                # I just need to get the API to stop randomly failing...
                print(i)
                temp_unavail_from = REMIT[REMIT["Fuel type"] == i].groupby("SP_s")["Unavailable MW"].sum().to_dict()
                temp_unavail_to = REMIT[REMIT["Fuel type"] == i].groupby("SP_e")["Unavailable MW"].sum().to_dict()
                
                # unavailable MWs from and unavailable MWs to (ie times they're unavailable from)
                df[f"{i} u_MW_from"] = df["Start time"].map(temp_unavail_from).fillna(0)
                df[f"{i} u_MW_to"] = df["End time"].map(temp_unavail_to).fillna(0)
                
                df[f"d_{i}"] = df[f"{i} u_MW_from"] - df[f"{i} u_MW_to"] 
                
                df[f"{i} Unavailable MW"] = df[f"d_{i}"].cumsum()
                
                #del df[f"{i} u_MW_from"], df[f"{i} u_MW_to"], df[f"d_{i}"]
                
            
            df.set_index("Start time", inplace = True)
            
            print(df)
            
            return REMIT, df
        
        def REMIT_old(self, start_date: str, end_date: str, only_last_revision: bool = True,
                  profile_only: bool = False):
            
            REMIT_MRIDs = Data_Import().Elexon_query().Get_Data(ID = "remit/list/by-event", start_date = start_date,
                                                              end_date = end_date, REMIT = True, 
                                                              only_last_revision = only_last_revision,
                                                              profile_only = profile_only)
            
            #REMIT_IDs = REMIT_IDs.rename(columns = Data_Import().column_renames)
            
            REMIT_MRIDs = REMIT_MRIDs["mrid"].unique().tolist()
            
            print(f"Number of REMIT MRIDs: {len(REMIT_MRIDs)}")
            
            
            #REMIT = Data_Import().Elexon_query().Get_Data("/remit", REMIT_MRIDs = REMIT_MRIDs)
            REMIT = Data_Import().Elexon_query().Get_Data(ID = "/remit/search", REMIT_MRID = REMIT_MRIDs, 
                                                          TEST = False).reset_index()
            # MRID appears to give the most recent revision number, so I can just use what comes out of this
            REMIT = REMIT[["ID", "mrid", "Start time", "End time", "BMU ID", "Normal MW", "Available MW", 
                           "Unavailable MW", "Event type", "Status", 
                           "Revision", "Duration uncertainty"]]
            
            
            REMIT = REMIT.drop_duplicates(keep = "first")
            
            REMIT["Start time"] = pd.to_datetime(REMIT["Start time"])
            REMIT["End time"] = pd.to_datetime(REMIT["End time"])            
            
            fuel_type_dict = pd.read_csv("BMU Info.csv").set_index("BMU ID").to_dict()["Fuel type"]
            REMIT["Fuel type"] = REMIT["BMU ID"].map(fuel_type_dict)
            REMIT["Fuel type"] = REMIT["Fuel type"].where(~REMIT["BMU ID"].str.startswith("I_"), "Interconnector")
            
            
            """Needs to get a start SP and end SP column as a datetime"""
            
            # =============================================================================
            # Creates a SP start time            
            # =============================================================================
                        
            REMIT["SP_s"] = REMIT["Start time"].dt.date.astype(str)
            REMIT["SP_e"] = REMIT["SP_s"]
            
            SP1a = REMIT["SP_s"] + " " + REMIT["Start time"].dt.hour.astype(str)
            SP1b = REMIT["SP_s"] + " " + "0" + REMIT["Start time"].dt.hour.astype(str)
            
            REMIT["SP_s"] = SP1a.where(REMIT["Start time"].dt.hour > 9, SP1b)
            
            SP2a = REMIT["SP_s"] + ":00:00"
            SP2b = REMIT["SP_s"] + ":30:00"
            
            REMIT["SP_s"] = SP2a.where(REMIT["Start time"].dt.minute < 30, SP2b)
            
            
            print(REMIT[["BMU ID", "Start time", "SP_s"]])
            
            # =============================================================================
            # Creates a SP end time            
            # =============================================================================
            
            """
            Note this bit here isn't perfect. Ie if a unit comes online at 00:01:00, it will be
            classed as offline until 00:30:00. This is something which could be improved
            """
            # end times whose hours are below 10
            SP1a = REMIT["SP_e"] + " 0" + REMIT['End time'].dt.hour.astype(str) + ":"
            # end times whose hours are over 10
            SP1b = REMIT["SP_e"] + " " + REMIT['End time'].dt.hour.astype(str) + ":"
            
            REMIT["SP_e"] = SP1a.where((REMIT["End time"].dt.hour < 10), SP1b)
            
            SP2a = REMIT["SP_e"] + "00:00"
            SP2b = REMIT["SP_e"] + "30:00"
            
            REMIT["SP_e"] = SP2a.where((REMIT["End time"].dt.minute == 0) | (REMIT["End time"].dt.minute > 31), SP2b)
            
            
            REMIT["SP_s"] = pd.to_datetime(REMIT["SP_s"], format = "%Y-%m-%d %H:%M:%S")
            REMIT["SP_e"] = pd.to_datetime(REMIT["SP_e"], format = "%Y-%m-%d %H:%M:%S")
            
            """Now need to try sum the unavailable capacity between the periods here"""
            
            dr_start = pd.date_range(start = start_date, end = end_date, freq = "30min")
            dr_end = dr_start + pd.DateOffset(days = 1/48)
            
            
            df = pd.DataFrame(index = dr_start, data = dr_end).reset_index()
            df.rename(columns = {"index": "Start time", 0: "End time"}, inplace = True)
            
            """Groupby REMIT by start date, then do the same for end date, then you should be 
            able to just minus them from each other no? (ie amount coming online - amount going offline
                                                         should equal the total amount online?)"""
            
            for i in sorted(REMIT["Fuel type"].astype(str).unique().tolist()):
                
                # having checked this method on 19 Jan 2025, I'm happy with the results it's putting out
                # I just need to get the API to stop randomly failing...
                print(i)
                temp_unavail_from = REMIT[REMIT["Fuel type"] == i].groupby("SP_s")["Unavailable MW"].sum().to_dict()
                temp_unavail_to = REMIT[REMIT["Fuel type"] == i].groupby("SP_e")["Unavailable MW"].sum().to_dict()
                
                df[f"{i} u_MW_from"] = df["Start time"].map(temp_unavail_from).fillna(0)
                df[f"{i} u_MW_to"] = df["End time"].map(temp_unavail_to).fillna(0)
                
                df[f"d_{i}"] = df[f"{i} u_MW_from"] - df[f"{i} u_MW_to"] 
                
                df[f"{i} Unavailable MW"] = df[f"d_{i}"].cumsum()
                
            
            df.set_index("Start time", inplace = True)
            
            df.to_csv("Unavailable MW.csv")
            
            return REMIT, df
        
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


if __name__ == "__main__":
    start_time = datetime.now()
    R, U = Data_Import().Elexon_query().REMIT("2024-12-01", "2024-12-10")
    U.to_csv("Unavailable MW testing.csv")
    R.to_csv("Unavailable MW testing data.csv", index = False)
    
    print(f"Data collected in {datetime.now() - start_time}")
else:
    pass