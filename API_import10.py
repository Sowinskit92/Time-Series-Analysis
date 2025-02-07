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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import webbrowser


pd.set_option("display.max_columns", None)


"""
The mrID one works well but there's too many mrIDs for it to be feasible, Elexon just stops you
from querying it

So I'm thinking about going back to the ID list but making sure the onlylastrevision argument is True

But I'm going to overhaul the REMIT function as that's confusing now

"""

def figures_to_html(figs, filename): #converts inputted figures to a chrome tab and opens it (figs must be a list, filename is a string)
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    
    webbrowser.open_new_tab(filename)


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
                               "fuelType": "Fuel type", "psrType": "Fuel type",
                               
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
                               "publishTime": "Published time",
                               
                               "transmissionSystemDemand": "Transmission demand",
                               "nationalDemand": "National demand"
                               
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
                     REMIT_IDs: list = None, TEST: bool = False, Other = None):
            
            def Elexon_Data_Query(URL: str, start_date: str = None, end_date: str = None,
                                  API_from_joiner: str = "from", API_to_joiner: str = "to", 
                                  fuel_types: list = None, REMIT: bool = False, 
                                  only_last_revision: bool = False, profile_only: bool = False,
                                  REMIT_IDs: list = None, TEST: bool = False, Other = None):
                
                # =============================================================================
                # This function queries the Elexon API       
                # =============================================================================
                
                def Update_URL(URL: str, start_date: str = None, end_date: str = None,
                               API_from_joiner: str = "from", API_to_joiner: str = "to", 
                               fuel_types: list = None, REMIT: bool = False, 
                               only_last_revision: bool = True, profile_only: bool = False,
                               REMIT_IDs: list = None, TEST: bool = False, Other: str = None) -> str:
                    
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
                    
                    # other is a string which needs to be added to the end of an API query
                    if Other == None:
                        pass
                    else:
                        URL = URL + f"&{Other}"
                    
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
                                            TEST = TEST, Other = Other)
                    
                    
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
                                                TEST = TEST, Other = Other)
                        
                        # aims to deal with errors in the request
                        session = requests.Session()
                        retry = Retry(connect=3, backoff_factor=0.5)
                        adapter = HTTPAdapter(max_retries=retry)
                        session.mount('http://', adapter)
                        session.mount('https://', adapter)
                        
                        
                        if i == 0:
                            data = session.get(URL_string).json()
                            try:
                                df = pd.json_normalize(data, record_path = "data")
                            except:
                                df = pd.json_normalize(data)
                        else:
                            data_temp = session.get(URL_string).json()
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
                                   profile_only = profile_only, REMIT_IDs = REMIT_IDs, TEST = TEST, 
                                   Other = Other)
            
        
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
                                                       only_last_revision = only_last_revision)
            IDs = IDs["id"].unique().tolist()
            REMIT = Data_Import().Elexon_query().Get_Data(ID = "/remit", REMIT_IDs = IDs, 
                                                          TEST = False)
            REMIT = REMIT.rename(columns = Data_Import().column_renames)
            
            # removes data we don't care about
            # we don't care about Dismissed ones because they didn't happen
            # we don't care about Inactive ones because they're updating something that happened
            # in the past, so this info wouldn't have been available to the market so wouldn't effect 
            # wholesale prices
            #REMIT = REMIT[REMIT["Status"] == "Active"]
            
            # doing the above didn't fix the problem, so I'd rather keep them and then it'd be
            # an accurate reflection of unavailable generation on the system at any point
            
            REMIT = REMIT[REMIT["Status"] != "Dismissed"]
            
            fuel_type_dict = pd.read_csv("BMU Info.csv").set_index("BMU ID").to_dict()["Fuel type"]
            REMIT["Fuel type"] = REMIT["BMU ID"].map(fuel_type_dict)
            REMIT["Fuel type"] = REMIT["Fuel type"].where(~REMIT["BMU ID"].str.startswith("I_"), "INTERCONNECTOR")
            
            REMIT["t"] = "INTERCONNECTOR (IMPORT)"
            REMIT["Fuel type"] = REMIT["t"].where((REMIT["Fuel type"] == "INTERCONNECTOR") & 
                                                  (REMIT["BMU ID"].astype(str).str[4] == "G"), REMIT["Fuel type"].astype(str))
            
            REMIT["t"] = "INTERCONNECTOR (EXPORT)"
            REMIT["Fuel type"] = REMIT["t"].where((REMIT["Fuel type"] == "INTERCONNECTOR") & 
                                                  (REMIT["BMU ID"].astype(str).str[4] == "D"), REMIT["Fuel type"].astype(str))
            
            del REMIT["t"]
            
            
            # I'm not deleting the column as there are occaisions where it gives the unavailable MW
            # without giving the available MWs
            
            REMIT["Unavailable MW"] = REMIT["Unavailable MW"].where((REMIT["Available MW"].isna()) |
                                                                    (REMIT["Normal MW"].isna()), 
                                                                    REMIT["Normal MW"] - REMIT["Available MW"])
            
            #t_cols = ["Published time", "Created time", "Start time", "End time"]
            t_cols = ["Created time", "Start time", "End time"]
            
            for i in t_cols:
                REMIT[i] = pd.to_datetime(REMIT[i]).dt.tz_localize(None)
            
            
            
            REMIT = REMIT[t_cols + ["BMU ID", "Fuel type", "Status", "Normal MW", "Available MW", "Unavailable MW"]]
            
            
            
            def Remove_Overlaps(REMIT, start_date, end_date):
                # =============================================================================
                # Function will iterate through BMU IDs to remove overlaps in the data           
                # =============================================================================
                
                
                dr_start = pd.date_range(start = start_date, end = end_date, freq = "30min")
                
                dr_end = dr_start + pd.DateOffset(days = (1/48))
                REMIT["SP_s"] = REMIT["Start time"].dt.round("30min").dt.tz_localize(None)
                REMIT["SP_e"] = REMIT["End time"].dt.round("30min").dt.tz_localize(None)
                
                df = pd.DataFrame(index = dr_start, data = dr_end).reset_index()
                df.rename(columns = {"index": "Start time", 0: "End time"}, inplace = True)
                
                
                
                IDs = REMIT["BMU ID"].unique().tolist()
                
                st = datetime.now()
                
                for i, j in enumerate(IDs):
                    print(f"{i + 1}/{len(IDs)}: {j}")
                    #print(REMIT)
                    
                    # Creates unit's remit (ur)
                    # sorts dataframe so the most recently created start date is nearer the top of the df
                    ur = REMIT[REMIT["BMU ID"] == j].sort_values(by = ["SP_s", "SP_e", "Created time"])
                    
                    # removes data where there's overlapping entries, keeps most recently created ones
                    ur = ur.drop_duplicates(subset = ["BMU ID", "SP_s", "SP_e", "Unavailable MW"], keep = "last")
                    '''
                    Overlaps can occur when:
                        1. An entry has an end date less than the one before it
                        2. An entry has a start date greater than the one before it
                        
                    
                    
                    temp["Overlap"] = 1
                    
                    # this line gets the first overlapping entry, which is the one we want to keep
                    temp["Overlap"] = temp["Overlap"].where(temp["SP_s"].shift(-1) < temp["SP_e"], 0)
                    # this line sets the next instance of the overlap to -1 (so we can remove it later)
                    temp["Overlap"] = temp["Overlap"].where(temp["Overlap"].shift(1) != 1, -1*temp["Overlap"].shift(1))
                    # this checks the one below any which has overlap at -1
                    temp["Overlap"] = temp["Overlap"].shift(1).where((temp["Overlap"].shift(1) == -1) & 
                                                                     (temp["SP_s"].shift(-2) < temp["SP_e"]), temp["Overlap"])
                    
                    '''
                    
                    unit_st = ur.set_index("SP_s").to_dict()["Unavailable MW"]
                    unit_et = ur.set_index("SP_e").to_dict()["Unavailable MW"]
                    
                    st = ur["SP_s"].min()
                    et = ur["SP_e"].max()
                    
                    # starts a temporary date range based off the unit's REMIT entries
                    temp_dr = pd.date_range(start = st, end = et, freq = "30min")
                    temp = pd.DataFrame(index = temp_dr)
                    temp["U_MW_s"] = temp.index.map(unit_st)
                    #temp["U_MW_s"] = temp["U_MW_s"].ffill()
                    temp["U_MW_e"] = temp.index.map(unit_et)
                    
                    # I'm just doing this bit so I can check things are working as they should be
                    x = ["T_CARR-1", "T_CARR-2", "E_ROARB-1"]
                    
                    if j  == "T_MEDP-1":
                        ur = ur[["SP_s", "SP_e", "Unavailable MW"]]
                        #temp = temp[["Created time", "SP_s", "SP_e", "Overlap"]]
                        #print(temp.stack())
                        #print(temp.rolling(window = 2))
                        print(ur)
                        print(temp.dropna(how = "all"))
                        temp.to_csv("T_MEDP-1 test.csv")
                        
                        sys.exit()
                    
                    
                sys.exit()
                
                print(f"Iterated through all BMU IDs in {datetime.now() - st}s")
                return REMIT
                    
            
            Remove_Overlaps(REMIT, start_date, end_date)
            sys.exit()
            """
            This bit pre-processes the data to keep on the entries we care about before it's
            then aggregated up by time period and fuel type
            
            """
            
            
            # =============================================================================
            # Creates list of start/end dates of plants from data given, which will feed into
            # the cumulative sum so the code knows whether to include values or not
            # =============================================================================
            
            # plant closures
            pc = pd.read_csv("BMU Closure Dates.csv")
            pc["Closure date"] = pd.to_datetime(pc["Closure date"], format = "%d/%m/%Y %H:%M")
            pc = pc.set_index("BMU ID").to_dict()["Closure date"]
            
            
            # REMIT["Closure date"] = REMIT["BMU ID"].map(pc)
            
            
            # =============================================================================
            # Creates SP start and end times     
            # =============================================================================
    
            
            #REMIT = REMIT.groupby(["BMU ID", "Fuel type", "SP_s", "SP_e"])["Unavailable MW"].max().reset_index()
            
            granularity = "1min"
            
            dr_start = pd.date_range(start = start_date, end = end_date, freq = granularity)
            
            dr_end = dr_start + pd.DateOffset(days = (1/(24*60)))
            REMIT["SP_s"] = REMIT["Start time"].dt.round("1min").dt.tz_localize(None)
            REMIT["SP_e"] = REMIT["End time"].dt.round("1min").dt.tz_localize(None)
            
            
            REMIT = REMIT.drop_duplicates(subset = ["BMU ID", "Fuel type", "Normal MW", "Available MW", "Unavailable MW", "SP_s", "SP_e"], keep = "first")
            
            
            
            df = pd.DataFrame(index = dr_start, data = dr_end).reset_index()
            df.rename(columns = {"index": "Start time", 0: "End time"}, inplace = True)
            
            # =============================================================================
            # Finds unavailable MW    
            # =============================================================================
                
            for i in sorted(REMIT["Fuel type"].astype(str).unique().tolist()):
                print(i)
                # sum of MW unavailable by fuel type by start time
                temp_unavail_from = REMIT[REMIT["Fuel type"] == i].groupby(["BMU ID", "SP_s", "SP_e"])["Unavailable MW"].max().reset_index()
                temp_unavail_from = temp_unavail_from.groupby("SP_s")["Unavailable MW"].sum().to_dict()
                # sum of MW unavailable by fuel type by end time
                temp_unavail_to = REMIT[REMIT["Fuel type"] == i].groupby(["BMU ID", "SP_s", "SP_e"])["Unavailable MW"].max().reset_index()
                temp_unavail_to = temp_unavail_to.groupby("SP_e")["Unavailable MW"].sum().to_dict()
                
                
                # unavailable MWs from and unavailable MWs to (ie times they're unavailable from)
                df[f"{i} u_MW_from"] = df["Start time"].map(temp_unavail_from).fillna(0)
                df[f"{i} u_MW_to"] = df["End time"].map(temp_unavail_to).fillna(0)
                
                # creates net change in unavailable MW column for each fuel type, making
                # sure that units which are only offline for <1 period are still included in 
                # the net difference for that period
                
                df[f"d_{i}"] = (df[f"{i} u_MW_to"] - df[f"{i} u_MW_from"]).where(df[f"{i} u_MW_to"] != df[f"{i} u_MW_from"], 0)
                
                
                new_column_title = f"Unavailable {i}"
                # finds cumulative total capacity offline
                df[new_column_title] = df[f"d_{i}"].cumsum()
                df[new_column_title] = (df[new_column_title] + df[f"{i} u_MW_from"].mul(-1)).where((df[f"{i} u_MW_from"] == df[f"{i} u_MW_to"]) &
                                                                                                   (df[f"{i} u_MW_from"] != 0), df[new_column_title])
                df[new_column_title] = df[new_column_title].round()
                
                
                
            df = df.set_index("Start time")
            print(df[[i for i in df.columns.tolist() if "unavailable" in i.lower()]])
            
            # would be smart to graph the % of normal capacity being shown for units to see where
            # things are going wrong
            
            
            CCGT_units = REMIT[REMIT["Fuel type"] == "CCGT"]["BMU ID"].unique().tolist()
            
            
            """
            Below works with the testing I've done so far, and doesn't return any unavailable MW
            which is greater than its rated capacity
            """
            
            def REMIT_by_Units(Units, REMIT, ):
            
                granularity = "1min"
                
                dr_start = pd.date_range(start = start_date, end = end_date, freq = granularity)
            
                df_test = pd.DataFrame(index = dr_start).reset_index()
                df_test.rename(columns = {"index": "Time"}, inplace = True)
                
                # capacity graph
                cg = go.Figure()
                
                for i in Units:
                    print(i)
                    
                    # unavailable MW
                    test_st = REMIT[REMIT["BMU ID"] == i].groupby("Start time")["Unavailable MW"].sum().mul(-1).to_dict()
                    # online MW
                    test_et = REMIT[REMIT["BMU ID"] == i].groupby("End time")["Unavailable MW"].sum().to_dict()
                    
                    #print(REMIT[REMIT["BMU ID"] == i])
                    #print(REMIT[REMIT["BMU ID"] == i].groupby("Start time")["Unavailable MW"].sum().mul(-1))
                    #print(REMIT[REMIT["BMU ID"] == i].groupby("End time")["Unavailable MW"].sum())
                    
                    # max capacity
                    max_MW = REMIT[REMIT["BMU ID"] == i]["Normal MW"].max()
                    
                    df_test[i] = df_test["Time"].map(test_st)
                    df_test[i] = df_test[i] + df_test["Time"].map(test_et)
                    df_test[i] = df_test[i].ffill()
                    df_test[f"{i} %"] = df_test[i].div(max_MW)
                    
                    if df_test[f"{i} %"].abs().max() > 1:
                        cg.add_trace(go.Scatter(x = df_test["Time"], y = df_test[f"{i} %"], name = i))
                
                return df_test
            
            
            
            print(REMIT[REMIT["BMU ID"] == "T_DIDCB6"][["SP_s", "SP_e", "Unavailable MW", "Created time"]])
            #print(REMIT[REMIT["BMU ID"] == "T_DIDCB6"][["SP_s", "SP_e", "t", "Overlapping", "Unavailable MW", "Created time"]])
            sys.exit()
            
            df_test.to_csv("Unavailable CCGT units.csv", index = False)
            REMIT.to_csv("REMIT data new4.csv", index = False)
            cg.update_layout(hovermode = "x unified", title = "Unavailable CCGT MW testing")
            figures_to_html([cg], f"Unavaiable CCGT MW testing.html")
            
            sys.exit()
        
        def DA_Demand_data(self, start_date: str):
            
            df = Data_Import().Elexon_query().Get_Data(ID = "forecast/demand/day-ahead/history", start_date = start_date,
                                                        API_from_joiner = "publishTime")
            
            df = df.rename(columns = Data_Import().column_renames)
            df["Start time"] = pd.to_datetime(df["Start time"])
            
            df = df[df["Date"].astype(str) == start_date]
            
            del df["boundary"], df["Published time"]
            
            return df
        
        def DA_Wind_data(self, start_date: str):
            
            # should only do one day at a time to match the DA_Demand_data function
            # code below creates an end_date for the API query, but df gets filtered to make sure
            # it's only showing the DA values
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + relativedelta(days = 1)
            end_date = datetime.strftime(end_date, "%Y-%m-%d")
            
            
            
            df = Data_Import().Elexon_query().Get_Data(ID = "/forecast/generation/wind-and-solar/day-ahead", 
                                                       start_date = start_date,
                                                       end_date = end_date, 
                                                       Other = "processType=all")
            df = df.rename(columns = Data_Import().column_renames)
            
            # filters df to make sure it's just got the values we care about
            df = df[(df["processType"] == "Day ahead") & (df["Date"].astype(str) == start_date)].reset_index()
            df["Start time"] = pd.to_datetime(df["Start time"])
            
            
            df = df[["Start time", "Date", "SP", "businessType", "quantity"]]
            df = df.rename(columns = {"quantity": "MW", "businessType": "Fuel type"}) # haven't included in main rename dict as other datasets have a 'quantity' column which aren't MW
            
            
            # below removes the onshore/offshore component of wind, and treats it as just wind
            df["Fuel type"] = df["Fuel type"].str.replace(" generation", "")
            df = df.groupby(["Start time", "Date", "SP", "Fuel type"])["MW"].sum().reset_index()
            
            df = df.sort_values(by = "Start time", ascending = True)
            
            print(df)
            return df
            
        def WA_Demand_data(self, start_date: str):
            
            # turns out elexon don't publish forecast demand figures down to SP
            # beyond the DA so I'll have to create my own forecast I think
            # temperature, time of day, day of week, month of year (not sure where I'd get the temperature forecast from though)
        
        
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
    
    
    
    df = Data_Import().Elexon_query().DA_Wind_data(start_date = "2024-01-01")
    
    
    sys.exit()
    REMIT, df = Data_Import().Elexon_query().REMIT("2024-12-01", "2024-12-31")
    
    #REMIT.to_csv("REMIT data new3.csv", index = False)
    #df.to_csv("REMIT outputs new3.csv")
    
    sys.exit()
    
    raise TypeError("Why am I getting positive available MWs for CCGT when I run it from Jan-23 to Dec-24 but not Jan-24 to Dec-24?")
    raise TypeError("The new closure date feature doesn't work yet! It's still showing COAL as having active capacity")
    
    """
    Issues with REMIT unavailable MW:
        1. (sorted) The d_{i} column was the wrong way round (ie units coming online would decrease available MW)
        2. When a unit comes online before anything comes offline, it makes it seem as though
           capacity has increased, when in fact it was more because capacity was offline but
           the data didn't go far back enough. 
           
           A way to deal with this could be to 1. Gather much more historic data than is
           needed to run the model or 2. Set any positive d_{i} values to 0 if there's been
           no decrease in available generation of that fuel type in the data so far.
           
           I think the second option would be better in all honesty
           
       3. (sorted) Interconnector fuel types should be split by generation and demand (ie import/export)
       
       4. (sorted) From looking at the data, it looks like we only want to include the 'active' ones.
          It looks like ones which are 'dismissed' have been cancelled or superceded by an
          'active' one
          
          Although saying that, T_STAY-2 doesn't seem to match this. It had some dismissed/inactive
          REMIT messages but its FPN was 0 during these times. However looking into it more this
          could be from the fact it just wasn't generating due to high wind
          
          Looks like we just want to omit any which were 'dismissed'. Needs more research though.
          
          Doesn't make sense, T_SIZB-2 was on outage for its entire capacity according to REMIT,
          from 11 October 2024 to 3 December 2024, but it's FPN + metered output shows it was generating
          consistently from 1 December but there's no REMIT update?
          
      5. (sorted) The 'Unavailable MW' that Elexon issue are awful. They list the normal capacity as something
         then the available capacity as something but the unavailable MW is sometimes completely
         unrelated to either of those. I need to make my own
         
     6. (sorted) In periods where a unit goes offline and then comes online, this isn't reflected.
        Eg if 620MW comes offline at the beginning of period 1 but comes online again at the
        end, this isn't reflected as having 620MW offline
        
    7. It now doesn't take into account when units come offline and somehow I'm still ending
       up with unavailable MWs which are greater than 0.
       
       It looks like it's because there's overlapping entries which have similar capacities.
       I might have to do this by BMU...?
       
   8. (sorted) Some values for T_DIDCB6 disappearing in Feb-24 which we needed
   
   9. I need to deal with REMIT entries which overlap in their start/end times (ie only include one of them)
   
        - one option could be to cap the values at the maximum capacity if it gets exceeded, but 
          I don't really like this as it will clear up any other mistakes in the data that might
          need to be sorted out. Also the unit may not ever have its full capacity offline during
          a period, instead it might have multiple overlapping ones which togehter sum to greater than
          its capacity. So doing this would give incorrect data
          
          
          - maybe find the overlapping ones then iterate through them to update the start/end dates
            and capacities off/online. If the data is sorted by SP_s, you can check if the overlaps
            are part of the same window of time or whether they're separate, then the start/end times
            can be updated?

        
                       
         
   """
   
    
    print(f"Data collected in {datetime.now() - start_time}")
else:
    pass