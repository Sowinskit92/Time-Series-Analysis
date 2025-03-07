# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:18:52 2025

@author: Tim.Sowinski
"""

from CI_data_imports21 import *

# =============================================================================
# In this script, I'm changing over the single 'fuel type' column into a 
# boolean column per fuel type, as it appears some units show battery
# and demand tendancies at the same time
# =============================================================================


st = datetime.now()
pd.set_option("display.max_columns", None)
#pd.reset_option('display.max_columns')

# Stage 1 is checking rate data from ELexon
Stage_1 = True

# Stage 2 is checking the physical data from the server
Stage_2 = True


# list of BMUs I know have some good data to test the methodology out on
test_list = ["2__ALOND002", "2__AFLEX004", "2__AFLEX001", "2__AECOT003"]

# if True, replaces BMU_list with test_list
TEST = False


# =============================================================================
# This script is designed back-date fuel types (not be run each day)
# =============================================================================

def Data_Gatherer():
    # this function gathers the necessary physical data from the server
    
    PN_file_name = "Physical data by BMU.pickle"
    
    print("Loading PN data...")
    if PN_file_name not in os.listdir():
        
        PN_IDs = {143: "PN", 
                 145: "MIL",
                 146: "MEL"}
        
        query_string = """

        ;with bmuIds as
         (select
          BMUnitID,
          Elexon_BMUnitID
         from
          meta.tblBMUnit
         where
          left(Elexon_BMUnitID,3) in ('2__', 'V__')
          and right(Elexon_BMUnitID,3) <> '000'
         )
        SELECT
            pd.[SettlementDate]
            ,pd.[DataDescriptionID]
            ,bmu.Elexon_BMUnitID
            ,min(pd.[LevelTo]) minLevelTo
         ,max(pd.[LevelTo]) maxLevelTo
        FROM
         [CI_DL1].[PowerSystem].[tblPhysicalData] pd
         inner join
          bmuIds bmu on bmu.BMUnitID = pd.BMUnitID
        WHERE
         pd.DataDescriptionID in (143,145,146)
        group by
         pd.[SettlementDate]
            ,pd.[DataDescriptionID]
            ,bmu.Elexon_BMUnitID


        """
        
        df = Data_Import().SQL_query().Custom_query(query_string)
        df = df.sort_values(by = ["Date", "BMU ID"], ascending = [True, True]).reset_index(drop = True)
        df["DataDescription"] = df["DataDescriptionID"].map(PN_IDs)
        
        df.to_pickle(PN_file_name)
    else:
        df = pd.read_pickle(PN_file_name)
    
    return df


def RURD_Battery_Finder_old(BMU_IDs: list, date_from: str, date_to: str):
    
    if isinstance(BMU_IDs, list):
        pass
    else:
        raise ImportError(f"{BMU_IDs} needs to be input as a list to gather run-up and run-down rates. Please review and try again")
    
    # creates time series of daily granularity to pull the values into
    date_from = date_from
    date_to = str(datetime.now().date() - relativedelta(days = 1))
    
    ts = pd.date_range("2019-01-01", date_to).date
    
    
    # creates master df to put a time series of all unit fuel types in
    fuel_type_df = pd.DataFrame(index = ts)
    
    df = pd.DataFrame(columns = ["Date", "BMU ID", "dataset", "rate1", "rate2", "rate3", "ID", "max_rate"])
    print(df)
    # separate fuel_type_df used in each iteration
    fuel_type_df_temp = pd.DataFrame(index = ts)
    
    for i, ID in enumerate(BMU_IDs):
        print(f"{(i + 1)/len(BMU_IDs)*100:.2f}%")
        # small, test df to see if there's any data. If there's not the code will skip this BMU ID
        RURD = Data_Import().Elexon_query(message = False).RURD_data(date_from = str(datetime.now().date() - relativedelta(days = 1)),
                                                                     date_to = date_to, BMU_ID = ID)
        if RURD.empty:
            # skips BMU ID if there's no RURD rate data
            pass
        else:
            RURD = Data_Import().Elexon_query().RURD_data(date_from = date_from, date_to = date_to, BMU_ID = ID)
            
            RURD["Date"] = pd.to_datetime(RURD["Date"]).dt.date
            RURD["ID"] = RURD["Date"].astype(str) + RURD["BMU ID"]
            
            # takes the max rate 
            RURD["max_rate"] = RURD[[i for i in RURD.columns.tolist() if "rate" in i.lower()]].max(axis = 1)
            
            df = pd.concat([df, RURD])
            
            print(df)
            
            sys.exit()
            
            RURD["Fuel type"] = "Battery"
            RURD["Fuel type"] = RURD["Fuel type"].where(RURD["max_rate"] == 999, "Unidentified")
            
            
            ft_map = RURD.set_index("Date").to_dict()["Fuel type"]
            
            
            
            fuel_type_df_temp["BMU ID"] = ID
            fuel_type_df_temp["Fuel type"] = fuel_type_df_temp.index.map(ft_map)
            
            fuel_type_df_temp["Fuel type"] = fuel_type_df_temp["Fuel type"].ffill()
            
            fuel_type_df_temp["Battery"] = 1
            fuel_type_df_temp["Battery"] = fuel_type_df_temp["Battery"].where(fuel_type_df_temp["Fuel type"] == "Battery", 
                                                                              0) 
            
            del fuel_type_df_temp["Fuel type"]
            
            # adds to master df
            fuel_type_df = pd.concat([fuel_type_df, fuel_type_df_temp])
            fuel_type_df = fuel_type_df[~fuel_type_df["BMU ID"].isna()] # removes na fuel type entries
            
            
    return fuel_type_df


def RURD_Battery_Finder(BMU_IDs: list, date_from: str, date_to: str):
    
    # =============================================================================
    # I've got rid of the code to determine a battery for now just so I can gather
    # the data
    # =============================================================================
    
    file_name_RURD = "RURD data by BMU.pickle"
    
    
    if isinstance(BMU_IDs, list):
        pass
    else:
        raise ImportError(f"{BMU_IDs} needs to be input as a list to gather run-up and run-down rates. Please review and try again")
    
    
    if file_name_RURD not in os.listdir():
    
        # creates time series of daily granularity to pull the values into
        date_from = date_from
        date_to = str(datetime.now().date() - relativedelta(days = 1))
        
        cols = ["Date", "BMU ID", "dataset", "rate1", "rate2", "rate3", "ID", "max_rate"]
        
        df = pd.DataFrame(columns = cols)
        
        
        for i, ID in enumerate(BMU_IDs):
            print(f"{(i + 1)/len(BMU_IDs)*100:.2f}%")
            # small, test df to see if there's any data. If there's not the code will skip this BMU ID
            RURD = Data_Import().Elexon_query(message = False).RURD_data(date_from = str(datetime.now().date() - relativedelta(days = 1)),
                                                                         date_to = date_to, BMU_ID = ID)
            if RURD.empty:
                # skips BMU ID if there's no RURD rate data
                pass
            else:
                RURD = Data_Import().Elexon_query().RURD_data(date_from = date_from, date_to = date_to, BMU_ID = ID)
                
                RURD["Date"] = pd.to_datetime(RURD["Date"]).dt.date
                RURD["ID"] = RURD["Date"].astype(str) + RURD["BMU ID"]
                
                # takes the max rate 
                RURD["max_rate"] = RURD[[i for i in RURD.columns.tolist() if "rate" in i.lower()]].max(axis = 1)
                
                RURD = RURD[cols]
                df = pd.concat([df, RURD]).reset_index(drop = True)
                
                print(df)
        
        df.to_pickle(file_name_RURD)
        
    else:
        df = pd.read_pickle(file_name_RURD)
    
    # =============================================================================
    # Goes through data and determines if it's a battery or not
    # =============================================================================
    
    df = df.groupby("ID")[["max_rate"]].max()
    
    df["Battery"] = 1
    df["Battery"] = df["Battery"].where(df["max_rate"] == 999, 0)
    
    
    return df
            
    
    
def PD_Demand_Finder_old(fuel_type_df, BMU_IDs: list, date_from: str, date_to: str):
    # =============================================================================
    # Maximum value of LevelTo for MIL is 0 (but LevelFrom can be > 0)
    # Minimum value of LevelTo for MEL is 0 (but LevelFrom can be < 0)
    # Therefore, makes sense to use LevelTo not LevelFrom
    # ----------------------------------------------------------------------------
    # Conditions to be classed as a 'pure' demand unit over a period:
    #   - MEL == 0
    #   - MIL <= 0
    #   - FPN <= 0
    #
    # Could be a mixed fuel type unit if these aren't met
    # =============================================================================
    
    #BMU_IDs = fuel_type_df["BMU ID"].unique().tolist()
    #raise ImportError("Currently, the PD_demand_Finder only finds the data for those which have dynamic data submitted. Do I want this? Or should I keep them all?")
    #fuel_type_df = fuel_type_df.reset_index().set_index(["index", "BMU ID"])
    
    ts = pd.date_range(date_from, date_to).date
    
    #fuel_type_df.to_csv("FT finder demand test 3.1.csv")
    
    # sets initial column value so the code doesn't overwrite itself later on 
    fuel_type_df["Demand"] = 0
    
    for i, ID in enumerate(BMU_IDs):
        
        fuel_type_df_temp = pd.DataFrame(index = ts)
        
        data = ["MIL", "MEL", "PN"]
        df = Data_Import().SQL_query().Physical_data(date_from = date_from, date_to = date_to,
                                                     data = data, BMU_ID = ID)
        
        df["Date"] = pd.to_datetime(df["TimeTo"]).dt.date
        
        
        # sets up the temporary fuel_type_df df to the master one
        fuel_type_df_temp["BMU ID"] = ID
        for j in data:
            _max = df[df["Description"] == j].groupby("Date")["LevelTo"].max().to_dict()
            _min = df[df["Description"] == j].groupby("Date")["LevelTo"].min().to_dict()
            
            fuel_type_df_temp[f"max_{j}"] = fuel_type_df_temp.index.map(_max)
            fuel_type_df_temp[f"min_{j}"] = fuel_type_df_temp.index.map(_min)
        
        """
        To be a 'pure' demand unit:
            - MEL cannot be greater than 0 (for reference, submitted MEL values cannot be less than 0)
            - MIL cannot exceed 0
            - FPNs must be less than or equal to 0
                - changing this to max_FPN < 0, as I think this is more reflective of demand
        """
        
        fuel_type_df_temp["Demand"] = 1
        fuel_type_df_temp["Demand"] = fuel_type_df_temp["Demand"].where((fuel_type_df_temp["max_MEL"] == 0) &
                                                                        (fuel_type_df_temp["max_MIL"] <= 0) &
                                                                        (fuel_type_df_temp["max_PN"] < 0), 
                                                                        0)
        
        ft_map = fuel_type_df_temp.to_dict()["Demand"]
        
        # updates master fuel_type_df with new fuel types
        fuel_type_df["Demand"] = fuel_type_df.index.map(ft_map).where((fuel_type_df["BMU ID"] == ID), 
                                                                      fuel_type_df["Demand"])
        #fuel_type_df["Demand"] = fuel_type_df["Demand"].fillna(0)
        
    #fuel_type_df.to_csv("FT_finder_demand_test8.csv")   
    return fuel_type_df


def PD_Demand_Finder():
    df = Data_Gatherer()
    df = pd.pivot_table(df, index = ["Date", "BMU ID"], columns = "DataDescription", values = ["minLevelTo", "maxLevelTo"])
    df.columns = [i[0].replace("LevelTo", "_") + i[1].replace("LevelTo", "") for i in df.columns]
    
    df = df.reset_index()
    df["ID"] = df["Date"].astype(str) + df["BMU ID"]
    df = df.set_index("ID")
    
    df["Demand"] = 1
    df["Demand"] = df["Demand"].where((df["max_MEL"] == 0) &
                                    (df["max_MIL"] <= 0) &
                                    (df["max_PN"] < 0), 
                                    0)
    
    #df = df[["Demand"]]
    
    return df


def Peaker_Finder(fuel_type_df, BMU_IDs: list, date_from: str, date_to: str):
    # =============================================================================
    # One potential option for this: sum up (PN/GC)*(hour) - use machine learning
    # algorithm to group IDs into categories based on this. (hour = hour + 0.5 if half hour)
    # - Effectively, each unit would get a score of how correlated their output is 
    # by the hour. 
    # Issue with this - later half hour periods would have a larger weight (not ideal as could mean
    # a unit is misclassified because it generates a reasonable amount over some periods but might not be a peaker)
    # ----------------------------------------------------------------------------
    # 
    # =============================================================================
    pass
    
    


# =============================================================================
# Gathers BMU data
# =============================================================================
if __name__ == "__main__":
    
    # =============================================================================
    # Gathers BMU IDs which need to be found
    # =============================================================================
    
    BMUs_all = Data_Import().SQL_query().BMU_data()
    
    # gathers only additional or secondary BMUs
    BMUs = BMUs_all[(BMUs_all["BMU ID"].str.startswith("2__")) | (BMUs_all["BMU ID"].str.startswith("V__"))]
    # removes any primary supplier units
    BMUs = BMUs[~BMUs["BMU ID"].str.endswith("000")]
    
    BMU_list = sorted(BMUs["BMU ID"].astype(str).unique().tolist())
    
    
    # =============================================================================
    # Gathers RURD data
    # =============================================================================
    
    # Should really group the SQL query by date and take the max value, as I don't
    # care about the values down to each SP, just by day as a fuel type isn't going
    # to change within day
    
    # sets the date to yesterday
    date_from = "2020-01-01"
    date_to = str(datetime.now().date() - relativedelta(days = 1))
    
    # =============================================================================
    # Checks RU/RD rates of the BMU to identify unit type
    # =============================================================================
    
    BESS = RURD_Battery_Finder(BMU_IDs = BMU_list, date_from = date_from, date_to = date_to)
    
    # =============================================================================
    # Checks historic MIL/MEL to identify unit type
    # =============================================================================
    
    DSR = PD_Demand_Finder()
    print(DSR[DSR["Demand"] == 1]["BMU ID"].unique())
    
    
    
    