import subprocess
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
import seaborn as sns
import pickle

"""
This modeul takes in the IPS data stored in 'test_dwnld/', smoothened OMNI data and Sun-spots data and generates the output .csv file for the time series prediction with:

i: index of observations
X: input features(12) of IPS and sunspot obs. for the past 32 quarter-day intervals (8-days)------12x32 columns
y: target OMNI data for the future 16 quarter-day intervals (4-days)-----------2x16columns
------------------------------------------------------------------------------------------------------------------------------------------------------------

The IPS data is staggered, therefore input X may not be present for any given time. The program generates time-series data where input X atleast has 
20 observations in the past 8-days.

"""



# The file format is as follows;

print("""
The IPS data has the following relevant columns
# Column
# -------------------------------------------------------------------------
# 1      SOURCE    Source name
# 2      YRMNDY    Date: Year-1900, Month, Day
# 3      UT        Universal time (Hour)
# 4      DIST      Radial distance, of P-point, from the sun (AU)
# 5      HLA       Heliocentric latitude of P-point (deg.)
# 6      HLO       Heliocentric longitude of P-point (deg.)
# 7      GLA       Heliographic latitude of P-point (deg.)
# 8      GLO       Heliographic latitude of P-point (deg.)
# 9      CARR      Carrington rotation number of P-point
# 10     V         Solar wind velocity (km/s)
#                  The value of -999 means that no velocity estimate is 
#                  available.
# 11     ER        The error in velocity estimation (km/s)
#                  The vale of -999 means either that only two station could
#                  be used to calculate the speed, or that no velocity estimate
#                  is available.
# 12     SC-INDX   Scintillation level (in arbitrary unit) observed 
#                  at either Fuji or Kiso station.
# -------------------------------------------------------------------------
Of these only a subset would be used for curation after preprocessing.\n\n
""")


# Column
# -------------------------------------------------------------------------
# 1      SOURCE    Source name
# 2      YRMNDY    Date: Year-1900, Month, Day
# 3      UT        Universal time (Hour)
# 4      DIST      Radial distance, of P-point, from the sun (AU)
# 5      HLA       Heliocentric latitude of P-point (deg.)
# 6      HLO       Heliocentric longitude of P-point (deg.)
# 7      GLA       Heliographic latitude of P-point (deg.)
# 8      GLO       Heliographic latitude of P-point (deg.)
# 9      CARR      Carrington rotation number of P-point
# 10     V         Solar wind velocity (km/s)
#                  The value of -999 means that no velocity estimate is 
#                  available.
# 11     ER        The error in velocity estimation (km/s)
#                  The vale of -999 means either that only two station could
#                  be used to calculate the speed, or that no velocity estimate
#                  is available.
# 12     SC-INDX   Scintillation level (in arbitrary unit) observed 
#                  at either Fuji or Kiso station.
# -------------------------------------------------------------------------

print(f"Preparing IPS data expects IPS data in forlder test_dwnld/.")

# -------------------------------------------------------------------------------------------------------------------------------------------------
vlist_directory = 'test_dwnld/'

url = 'https://stsw1.isee.nagoya-u.ac.jp/vlist/'
print(f"IPS data can be found in {url}.")

file_name = [str((x - 1900)%100) if len(str((x - 1900)%100))>1 else '0'+str((x - 1900)%100) for x in range(1983, 2025) ]

file_name = ['VLIST' + x for x in file_name]

print(f"File names expected {file_name}")
print(f"\nas seen in {url}.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# There were some extra coloumns, ignoring them at the momemnt.
columns_extra =['RA(B1950)_0', 'RA(B1950)_1', 'RA(B1950)_2', 'DC(B1950)_0',
                'DC(B1950)_1','DC(B1950)_2', 'RA(J2000)_0', 'RA(J2000)_1', 'RA(J2000)_2', 'DC(J2000)_0', 
                'DC(J2000)_1', 'DC(J2000)_2']
column_names = ['SOURCE', 'YRMNDY', 'UT', 'DIST', 'HLA',  'HLO', 'GLA', 'GLO', 'CARR', 
                'V', 'ER', 'SC-INDX', 'file']
column_names = [x.lower() for x in column_names]
print(f"Columns to be retained: {column_names}")
print(f"Extra columns to be dropped{columns_extra}")

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Here we concatenate files having read them one after the other as they ocuur in file_names.
# Note: everytime the index needs to redifined to start from where the previous file left off.
# An extra coloumn 'file' is added to keep track.

print("Making IPS DataFrame..")
df_test_1 = pd.DataFrame(columns=column_names)
for x in file_name:
    indx_strt = df_test_1.shape[0]
    df_test_0 = pd.read_csv(vlist_directory + x, sep='\s+', skipinitialspace=True, skiprows=8, header=None, 
                        names=column_names[0:-1], usecols=[y for y in range(len(column_names[0:-1]))])
    df_test_0.index = df_test_0.index + indx_strt
    df_test_0['file'] = [x for i in range(df_test_0.shape[0])]
    df_test_1 = pd.concat([df_test_1, df_test_0])  
print("IPS DataFrame made.")

# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
"""
In the rows where the error 'er' entry is -999 there was no space between the the entry for 'er' and 'v' (wind speed).
Therefore the combined reading of 'v' and 'er' was entered as a string in 'v'. 
Also some entries of 'v' where string instead of int.
Below we have a fucntion which can be applied row-wise to the data frame. 
"""
def v_err(row):
    if type(row.v) == str:
        if row.v[-4:] == '-999':
            row.v = int(row.v[0:-4])
            row['sc-indx'] = row.er
            row.er = -999
        else:
            row.v = int(row.v)
    return row


print(f"Formating v_err column")
# Implementing v_err 
df_test_1 = df_test_1.apply(v_err, axis=1)
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Implementing one-hot encoding for error 'er' values of -999 in new coloumn 'er_1'")
# Implementing one-hot encoding for error 'er' values of -999 in new coloumn 'er_1'
df_test_1['er_1'] = df_test_1.er.map(lambda x: 1 if x==-999 else 0)
df_test_1
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("'yrmndy' is read as an integer, so converting it into a string and adding required zeros for the first few years in the 2000s")
# The 'yrmndy' is read as an integer, so converting it into a string and adding required zeros
# for the first few years in the 2000s
def yr_mod(col):
    x = str(col)
    if len(str(x))<6:
        x = ''.join([ '0' for j in range(6 - len(x)) ]) + str(x) 
    else:
         x 
    return x

print("Implementing datetime stamp on 'yrmndy'")
# Implementing datetime stamp on 'yrmndy'
df_test_1['yrmndy'] = pd.to_datetime(df_test_1.yrmndy.map(yr_mod), format='%y%m%d')
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Copying to make changes in the copy to avoid running the whole file from the top
df_test_2 = df_test_1.copy()
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Implementing conversion to datetime by adding 'yrmndy' to it")
# Implementing conversion to datetime by adding yrmndy to it
df_test_2['ut'] = df_test_2.yrmndy + pd.to_timedelta(df_test_2.ut, unit='h')
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Sun Spots CSV 
# from https://www.sidc.be/SILSO/datafiles
# --------------------------------------------------------------------------------------------------------------------------------------------
# Filename: SN_d_tot_V2.0.csv
# Format: Comma Separated values (adapted for import in spreadsheets)
# The separator is the semicolon ';'.

# Contents:
# Column 1-3: Gregorian calendar date
# - Year
# - Month
# - Day
# Column 4: Date in fraction of year.
# Column 5: Daily total sunspot number. A value of -1 indicates that no number is available for that day (missing value).
# Column 6: Daily standard deviation of the input sunspot numbers from individual stations.
# Column 7: Number of observations used to compute the daily value.
# Column 8: Definitive/provisional indicator. '1' indicates that the value is definitive. '0' indicates that the value is still provisional.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Making sun_sopts DataFrame, expecting file 'SN_d_tot_V2.0.csv'")
sun_spots = pd.read_csv('SN_d_tot_V2.0.csv', sep=";", names=['year', 'month', 'day', 'date_frac', 'day_total', 'day_std', 'num' ,'d_p'])
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Implementing a 'yrmndy' column from 'year', 'month' and 'day', and dropping the latter columns")
# Implementing a 'yrmndy' column from 'year', 'month' and 'day', and dropping the latter columns 
sun_spots['yrmndy'] = pd.to_datetime(sun_spots[["year", "month", "day"]])
sun_spots.drop(columns=['year', 'month', 'day', 'date_frac'], inplace=True)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Editing -1 in 'day_total' and 'day_std' to reflect np.nan")
# Editing -1 in 'day_total' and 'day_std' to reflect np.nan
sun_spots['day_total'] = sun_spots.day_total.apply(lambda x: np.nan if x == -1 else x)
sun_spots['day_std'] = sun_spots.day_std.apply(lambda x: np.nan if x == -1 else x)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("""
Defining OMNI start date
This is the date from which Datta has currated the OMNI data w.r.t. smoothness
The official OMNI data starts from 1963
""")
# Defining OMNI start date
# This is the date from which Datta has currated the OMNI data w.r.t. smoothness
# The official OMNI data starts from 1963
omni_start_date_str = "1996-08-01 12:00:00"
omni_start_date_prv_mnth_str = "1996-07-01 12:00:00"
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Define sun_spots_recent from the start of the OMNI data")
# Define sun_spots_recent as the from the start the OMNI data
sun_spots_recent = sun_spots.loc[sun_spots.yrmndy >= pd.to_datetime(omni_start_date_prv_mnth_str)].copy()
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Conditions: er_1 = 0, er< 50,  0.25<dist<0.8

# Input: Last 4-8 days

# Criterion: Keep 64 obs

# Horizontal feat: subset of columns (timestamp, v, lo, la, etc )

# For every hour we need a table which has the following properties:
# 1. Columns as some selective columns of the database
# 2. A max of 32 or 64 rows filled with observations from the above mentioned hour till the past 6-days
# 3. The above observations from the past 6x24 hrs would be selected from total observations present in the 6x24hrs window
# 4. The selection has to sample observations in such a manner that the ones that are skipped have a chance of being sampled in the coming hours.
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print(" Conditions on IPS data: er_1 = 0, er < 50,  0.25 < dist < 0.8")
print("Defining conditions on the IPS data to be currated")
# Defining conditions on the IPS data to be currated
cond = (df_test_2.er_1==0) & (df_test_2.er < 50) & (df_test_2.dist < 0.8) & (df_test_2.dist > 0.25)

df_test_3 = df_test_2.loc[cond].copy()
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print(" delta_t is the number of days between the start of the IPS data and '2050-01-01'")
# delta_t is the number of days between the start of the IPS data and 2050-01-01
delta_t = (lambda x: x.days + x.seconds/(24*60*60))(pd.to_datetime('2050-01-01') - df_test_3.ut.min()) # no.of days from first obs to 2050-01-01
print(f"delta_t = {delta_t}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("""
Defining 'time' as 1e3*(1 - day_diff/delta_t) where day_diff = no.of days between 2050-01-01 and time of IPS obs in 'ut' column.
This time would be used for making the final data.
""")
# Defining 'time' as 1e3*(1 - day_diff/delta_t) where
# day_diff = no.of days between 2050-01-01 and time of IPS obs in 'ut' column.
# This time would be used for making the final data
df_test_3['time'] =  (1000 - (pd.to_datetime('2050-01-01') - df_test_3.ut).map(lambda x: 1000*(x.days + x.seconds/(24*60*60)))/delta_t).round(8)
# df_test_2['time'] =  (pd.to_datetime('2050-01-01') - df_test_2.ut).map(lambda x: x.seconds/(24*60*60)).round(8)
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Merge sun_spots_recent  to the left on 'yrmndy' column")
# Merge sun_spots_recent  to the left on 'yrmndy' column
df_test_3 = pd.merge(left=df_test_3, right=sun_spots_recent[['yrmndy', 'day_total']], on='yrmndy', how='left')
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("""
Note some time intervals the the format used in 'time' column
6 day interval in terms of units of the time col. i.e. 6x1000/delta_t
where delta_t is the no.of days between 2050-01-01 and first obs.
""")
# Note some time intervals the the format used in 'time' column
# 6 day interval in terms of units of the time col. i.e. 6x1000/delta_t
# where delta_t is the no.of days between 2050-01-01 and first obs.
delta_6 = 6*1e3/delta_t
delta_8 = 8*1e3/delta_t
print('6 days =',delta_6 , '8 days =',delta_8 , '.   1day =',(1)*1e3/delta_t, '.   1hr =',(1/24)*1e3/delta_t, '.   5min =',(1/(24*12))*1e3/delta_t)
round(delta_6, 5)
delta_1h = (1/24)*1e3/delta_t
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# OMNI start date in 'time' column format
omni_start_date = round((lambda x: 1000 - 1000*(x.days + x.seconds/(24*60*60))/delta_t)(pd.to_datetime('2050-01-01') - pd.to_datetime("1996-08-01 12:00:00")), 8)
print(f"OMNI start date is calibrated to {omni_start_date}")

print("Selecting IPS data from 8-days prior to start of omni_start_date")
# Selecting IPS data from 8-days prior to start of omni_start_date
df_4 = df_test_3.loc[df_test_3.time >= omni_start_date - delta_8] # IPS data set from 8days before start of omni data set
drop_cols = ['source', 'yrmndy', 'ut', 'file', 'er_1'] ## columns to be dropped during training
df_5 = df_4.drop(columns=drop_cols).copy()
df_5.reset_index(inplace=True)

print(f"Curated IPS data has columns: {list(df_5.columns)}.")

print("Saving curated IPS data to disk as 'curated_data/IPS_Sunspots_curated.csv'")
# Save df_5 to disk
os.makedirs("curated_data", exist_ok=True) 
df_5.to_csv("curated_data/IPS_Sunspots_curated.csv", index=False)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def find_ranked_er(time, time_delta):
    """
    Parameters:
    --------------------------------------------------
    time: time in df_5.time format
    time_delta: time interval in df_5.time format


    Returns:
    -------------------------------------------------
    np.array of ranked list of df_5 indices according to least error i.e. df_5.er
    """
    df = df_5.loc[(df_5.time <= time) & (df_5.time > time - time_delta)]
    if len(df) > 0:
        return df.er.sort_values().index
    else:
        return np.array([])
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def fill_bracket(time_0, time_1, intervals):
    """
    Params:
    -----------------------------------------------
    time_0: time in time formart of df_5
    time_1: < time_0
    intervals: total # equally spaced time intervals b/w time_0 and time_1

    Returns:
    -------------------------------------------------
    'intervals' many obs- with one obs of least error in each interval, as a pd.DataFrame().values.
    If no value is found in an interval, then its filled with the remainder set once all the intervals have been filled with the best within them 
    """
    time_delta = (time_0 - time_1)/intervals # size of each interval
    # print(time_delta)
    df = df_5.loc[(df_5.time <= time_0) & (df_5.time > time_1)].copy()

    
    if len(df) <= intervals:
        bracket = df
        # print('less')
    else:
        # print('more')
        # First fill in the intervals with the best obs from the same interval
        rest_obs_id = np.array([])
        empty_list = [] # list for intervals with no obs
        bracket = []   # to store obs rows
        for i in range(intervals):
            obs_id = find_ranked_er(time_0 - i*time_delta, time_delta)     ## get list of indices with obs in the interval ranked acc. to error i.e. df_5.er
            # print(i)
            if len(obs_id) != 0:
                bracket.append(df.loc[df.index==obs_id[0]].values.reshape((1,-1)).tolist())
                if len(obs_id) > 1:
                    obs_id = np.delete(obs_id, 0)
                    rest_obs_id = np.concatenate((rest_obs_id, obs_id))   # storing the ranked obs indices for filling unfilled intervals
                # break
            else:
                empty_list.append(i) # keeping track of empty intervals
    
        bracket = np.array(bracket)
        bracket = bracket.reshape((-1,len(df_5.columns))) 
        # print(bracket.shape)
        
        # Fill the rest of the intervals if any with the remainder of obs from other intervals
        rest_obs_id = rest_obs_id.astype(int)
        if len(rest_obs_id) > 0:
            for i, obs_j in zip(empty_list, rest_obs_id):
                # print(obs_j in list(df.index))
                bracket = np.concatenate((bracket, df.loc[df.index==obs_j].values.reshape((1,-1))))

    # Arranging the dataset according to time
    bracket = pd.DataFrame(bracket, columns=df.columns)
    bracket.drop(columns=['index'], inplace=True)
    bracket.sort_values('time', ascending=False, inplace=True)
    bracket.reset_index(inplace=True)
    bracket.drop(columns='index', inplace=True)
    del df
    return bracket
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
print(f"Preparing OMNI data: expects file omni_avg_normalised_smoothed_extened.csv")
print("""\n
Now we need to create the input data with 

target: as the omni values for every hour

input: as the 2d array (dataframe as np.array) consisting of the best obs in the past 8 days divivded into 32 intervals

here time is parametrized as follows:
time = 1000 - 1000*(#days b/w current time and 2050-01-01)/delta_t
delta_t = #days b/w (1983-03-25 04:18:00) and 2050-01-01\n
""")

# Now we need to create the input data with 

# target: as the omni values for every hour

# input: as the 2d array (dataframe as np.array) consisting of the best obs in the past 8 days divivded into 32 intervals

# here time is parametrized as follows:
# time = 1000 - 1000*(#days b/w current time and 2050-01-01)/delta_t
# delta_t = #days b/w (1983-03-25 04:18:00) and 2050-01-01

omni = pd.read_csv('omni_avg_normalised_smoothed_extened.csv')
# omni.head()
omni.rename(columns={'Unnamed: 0':'yrmndy_hr'}, inplace=True)
# omni.columns

print("Selecting omni columns and inserting a 'time' column similar to one in the curated IPS data")
# Selecting omni columns and inserting a 'time' column similar to one in the curated IPS data in df_5

omni_df = omni[['yrmndy_hr', 'swSpeed_Smth_0']].copy()

omni_df['yrmndy_hr'] = pd.to_datetime(omni_df.yrmndy_hr)

omni_df['time'] = (1000 - (pd.to_datetime('2050-01-01') - omni_df.yrmndy_hr).map(lambda x: 1000*(x.days + x.seconds/(24*60*60)))/delta_t).round(8)

omni_df.drop(columns=['yrmndy_hr'], inplace=True)

print(f"Curated OMNI data has columns: {list(omni_df.columns)}.")

print("Saving the curated OMNI data to disk as 'curated_data/OMNI_curated.csv'.")
# Saving the curated OMNI data to disk
omni_df.to_csv("curated_data/OMNI_curated.csv", index=False)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def make_training_data(ips_data, omni_data, min_input_len=20):
    """
    Construct training data as follows:
    Begining at every hour i of the omni_data construct
    the target y: 16 omni_data obs into the future begining at hour i i.e. omin_data 4 days into the future
    the input  x: best ips_data from 8-days into the past begining from the hour i as a pd.DataFrame().values
    
    
    The length of the omni_data controls the length of the output.
    If the input x generated doesn't have length of min_len_input then the data corresponding to the hour i is not considered.
    
    Params:
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    omni_data: omni data as pd.DataFrame with time as in time in df_5 and smoothed hourly solar wind speed.
    ips_data: ips pd.DataFrame() with relevant columns and time as above along with sun spot numbers for the relevant days.
    min_input_len: min number of ips_data in the past 8-days- if less, then the data point is skipped.

    Returns:
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    A list s.t. each row is a list [i, x, y] where 
    i: is the index starting from 0
    y: is the above target and 
    x: is the above (2dim with x.shape:(32, #selected columns from ips_data)) input
    
    A missing list containing rows [i, missed] where
    i: the index where no.of x data generated is not of length of 32
    missed: length of x data
    """
    
    out_data = [] 
    j = 0  # index 
    k = 0  # no.of samples skipped
    missing = []
    for i in range(len(omni_data) - 16):
        time = omni_data.iloc[i].time
        x_brckt = fill_bracket(time, time - delta_8, 32) # x_brckt has max len 32, it can be smaller
        x_brckt_len = len(x_brckt)
        # Do not make sample if x_brckt has len < 20
        if x_brckt_len < min_input_len:
            k = k + 1
            continue
        
        # adding an extra column in x for keeping track of the time of the input
        # this column has time as its entry for the first len(x_brckt) entries  
        # and then np.zeros for the remainding entries upto 32 if len(x_brckt) < 32
        # print(len(x_brckt))


        if x_brckt_len < 32:
            x_brckt_0 = pd.DataFrame(np.zeros(11*(32 - x_brckt_len)).reshape((32 - x_brckt_len), -1), columns=x_brckt.columns)
            x_brckt = pd.concat([x_brckt, x_brckt_0])
        # if len(x_brckt) == 32:
        #     time_0 = time*np.ones(32)
        # else:
        #     time_0 = np.concatenate((time*np.ones(len(x_brckt)), np.zeros(32 - len(x_brckt)) ))
        time_0 = time*np.ones(32)
        x_brckt['time_trgt'] = pd.Series(time_0) 
            
        # Uncomment line below to return a list with X and y as pd.DataFrames
        # out_data.append([j, x_brckt, omni_data.iloc[i: i + 16] ]) 
            
        out_data.append([j] + list(x_brckt.values.reshape(-1)) + list(omni_data.iloc[i: i+16].values.reshape(-1)))
        
        # Keep track of x_brckt when len < 32
        if x_brckt_len < 32:
            missing.append([j, x_brckt_len])
        
        j = j + 1
            
    print(f"{k} Data points skipped due to lack of atleast {min_input_len} IPS data points in the past 8-days.")
    if len(missing) > 0:
        print(pd.DataFrame(missing, columns=['id', 'missed']).describe().to_string())
    print(f"{j} Data points made.")
    return out_data, missing
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print(f"Generating column names in list clmns_data for the final data from curated IPS data:df_5 and curated OMNI data: omni_df")
# Generating column names in list clmns_data for the final data from curated IPS data:df_5 and curated OMNI data: omni_df

clmns_ips = list(df_5.columns)
clmns_ips.pop(0)
clmns_ips.append('time_trgt')
print(clmns_ips, len(clmns_ips))
clmns_input = []
for i in range(32):
    for clmn in clmns_ips:
        clmns_input.append(f"X_{clmn}_{i}")
# print(clmns_input)
clmns_omni = list(omni_df.columns)
clmns_target = []
for i in range(16):
    for clmn in clmns_omni:
        clmns_target.append(f"y_{clmn}_{i}")
clmns_data = ['idx'] + clmns_input + clmns_target
print("""\n
The final data has 1 + 12x32 + 2x16 = 417 columns where
1: corresponds to the index
12x32: corresponds to X input with 12 columns of the currated IPS data repeated for 32 obs.(8-days into past) with np.zeros used if no.of IPS obs. < 32
2x16: corresponds to OMNI smoothened wind speed and time for 16 obs.(4-days into future)

12 IPs columns correspond to the 11 mentioned previously and one denoting input time which same for all 32 data points.\n
""")
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n Making the final data\n")
# Making the final data 
full_out_1, missing = make_training_data(df_5, omni_df)
print("Final data made")
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------
print("Making DataFrame for missing data stats and storing to disk final_data/missing_df.csv")
# DataFrame for missing data stats and storing to disk
missing_df = pd.DataFrame(missing, columns=['id', 'missed'])
missing_df.id.describe()
os.makedirs("final_data", exist_ok=True) 
missing_df.to_csv('final_data/missing_df.csv',  index=False)

print("Converting final data into DataFrame and storing to disk as 'final_data/full_1_df.csv'")
# Converting final data into DataFrame and storing to disk
full_1_df = pd.DataFrame(full_out_1, columns=clmns_data)

print("Converting 'X_time_trgt_i' columns to 'X_time_trgt_i' - 'X_time_i' ")
print("Converting 'X_time_trgt_i' columns to 0.0 if 'X_time_trgt_{i} > 1' as this would imply the input rows 'X_..' are 0.0 i.e. no entries ")
for i in range(32):
    full_1_df[f'X_time_trgt_{i}'] = full_df[f'X_time_trgt_{i}'] - full_df[f"X_time_{i}"]
    full_1_df[f'X_time_trgt_{i}'] = full_df[f'X_time_trgt_{i}'].apply(lambda x: 0.0 if x > 1 else x )

def nan_maker(row):
    """
    Makes input rows 'X_..' np.nan if 'X_time_trgt_i' == 0.0
    """
    for i in range(32):
        if row[f"X_time_trgt_{i}"] == 0.0:
            for clm in X_clms:
                row[f"X_{clm}_{i}"] = np.nan
    return row
print("Making input entries in columns 'X_..' np.nan where 'X_time_trgt_i' = 0.0 ")
full_1_df = full_1_df.apply(nan_maker, axis=1)

print("Storing to disk final_data/missing_df.csv")
full_1_df.to_csv('final_data/full_1_df.csv', float_format='%.17g', index=False)
# ------------------------------------------------------------------------------------------------------------------------------------------------------