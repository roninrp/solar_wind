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
pd.set_option('display.float_format', lambda x: f'{x:.10f}')

"""
This module contains the class ips_omni_processor() which allows the making of input data from:

1. IPS data stored in a directory with yearly files labeled VLIST<yy> where <yy> is 01 for 2001 and 91 fro 1991.
2. Curated and smothened omni data used previously in encoder decoder nn
3. Sun spot data of daily total sun-spots


The data made consists -for a given instance of time, a pd.Dataframe with columns of the form [idx, {X}, {y}]

where idx: index
      {X}: X_{input_clmn}_{i}
              where input_clmn takes values ['dist', 'hla', 'hlo', 'gla', 'glo', 'carr', 'v', 'er', 'sc_indx', 'time', 'day_total']
              and i takes values range(32)
      {y}: y_swSpeed_Smth_0_{i}
              where i takes values range(32)


Begining at every hour i of the omni_data construct
        the target y: 16 omni_data obs into the future begining at hour i i.e. omin_data 4 days into the future
        the input  X: best ips_data from 8-days into the past begining from the hour i as a pd.DataFrame().values


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


The file format for IPS data is as follows;

Column
-------------------------------------------------------------------------
1      SOURCE    Source name
2      YRMNDY    Date: Year-1900, Month, Day
3      UT        Universal time (Hour)
4      DIST      Radial distance, of P-point, from the sun (AU)
5      HLA       Heliocentric latitude of P-point (deg.)
6      HLO       Heliocentric longitude of P-point (deg.)
7      GLA       Heliographic latitude of P-point (deg.)
8      GLO       Heliographic latitude of P-point (deg.)
9      CARR      Carrington rotation number of P-point
10     V         Solar wind velocity (km/s)
                 The value of -999 means that no velocity estimate is 
                 available.
11     ER        The error in velocity estimation (km/s)
                 The vale of -999 means either that only two station could
                 be used to calculate the speed, or that no velocity estimate
                 is available.
12     SC-INDX   Scintillation level (in arbitrary unit) observed 
                 at either Fuji or Kiso station.
-------------------------------------------------------------------------
"""

class ips_omni_processor():
    def __init__(self, ips_path, omni_path, sun_spot_path):
        self.ips_path = ips_path
        self.omni_path = omni_path
        self.sun_spot_path = sun_spot_path
        
        self.file_name = [str((x - 1900)%100) if len(str((x - 1900)%100))>1 else '0'+str((x - 1900)%100) for x in range(1983, 2025) ]

        self.file_name = ['VLIST' + x for x in self.file_name]
        print("IPS files found in folders: \n", self.file_name)
        
        
        self.column_names = ['SOURCE', 'YRMNDY', 'UT', 'DIST', 'HLA',  'HLO', 'GLA', 'GLO', 'CARR', 
                'V', 'ER', 'SC-INDX', 'file']
        self.column_names = [x.lower() for x in self.column_names]
        # Here we concatenate files having read them one after the other as they ocuur in file_names.
        # Note: everytime the index needs to redifined to start from where the previous file left off.
        # An extra coloumn 'file' is added to keep track.
        self.df_test_1 = pd.DataFrame(columns=self.column_names)
        # print(f"{self.df_test_1.shape}")
        
        print(f"Making sun-spots data....")
        self.sun_spots = pd.read_csv(self.sun_spot_path, sep=";", names=['year', 'month', 'day', 'date_frac', 'day_total', 'day_std', 'num' ,'d_p'])
        self.sun_spots_recent = self.sun_spots.copy() 
        # Implementing a 'yrmndy' column from 'year', 'month' and 'day', and dropping the latter columns 
        self.sun_spots['yrmndy'] = pd.to_datetime(self.sun_spots[["year", "month", "day"]])
        self.sun_spots.drop(columns=['year', 'month', 'day', 'date_frac'], inplace=True)
        # Editing -1 in 'day_total' and 'day_std' to reflect np.nan
        self.sun_spots['day_total'] = self.sun_spots.day_total.apply(lambda x: np.nan if x == -1 else x)
        self.sun_spots['day_std'] = self.sun_spots.day_std.apply(lambda x: np.nan if x == -1 else x)
        
        print(f"Making omni data ........")
        self.omni = pd.read_csv(omni_path)
        self.omni.rename(columns={'Unnamed: 0':'yrmndy_hr'}, inplace=True)
        self.omni_df = self.omni[['yrmndy_hr', 'swSpeed_Smth_0']].copy()
        self.omni_start_date_str = str(self.omni_df.iloc[0,0])
        self.omni_start_date_prv_mnth_str = "-".join([
            x 
            if i != 1 
            else (str(int(x) + 1) if len(str(int(x) + 1)) == 2 else "0"+str(int(x) + 1)) 
            for i, x in enumerate(str(self.omni_df.iloc[0,0]).split("-"))
        ])
        
        print(f"Taking sun spots data from one month prior to omni start date....")
        self.sun_spots_recent = self.sun_spots.loc[self.sun_spots.yrmndy >= pd.to_datetime(self.omni_start_date_prv_mnth_str)].copy()
        
        print(f"Formatting IPS data .........")
        self.df_test_2 = self.make_ips_df().copy()
        # print(f"{self.df_test_2.shape}")
        print(f"Conditioning ips data .........")
        self.cond = (self.df_test_2.er_1==0) & (self.df_test_2.er < 50) & (self.df_test_2.dist < 0.8) & (self.df_test_2.dist > 0.25)
        self.df_test_2 = self.df_test_2.loc[self.cond]
        self.delta_t = (lambda x: x.days + x.seconds/(24*60*60))(pd.to_datetime('2050-01-01') - self.df_test_2.ut.min())
        self.df_test_2['time'] =  (1000 - (pd.to_datetime('2050-01-01') - self.df_test_2.ut).map(lambda x: 1000*(x.days + x.seconds/(24*60*60)))/self.delta_t)
        self.df_test_2 = pd.merge(left=self.df_test_2, right=self.sun_spots_recent[['yrmndy', 'day_total']], on='yrmndy', how='left')
        
        
        self.omni_start_date = round((lambda x: 1000 - 1000*(x.days + x.seconds/(24*60*60))/self.delta_t)(pd.to_datetime('2050-01-01') - 
                                                                                                     pd.to_datetime(self.omni_start_date_str)), 8)
        print(f"OMNI start date is calibrated to {self.omni_start_date}")
        
        self.delta_6 = 6*1e3/self.delta_t
        self.delta_8 = 8*1e3/self.delta_t
        self.df_test_2 = self.df_test_2.loc[self.df_test_2.time >= self.omni_start_date - self.delta_8] # IPS data set from 8days before start of omni data set
        self.drop_cols = ['source', 'yrmndy', 'ut', 'file', 'er_1'] ## columns to be dropped during training
        self.df_5 = self.df_test_2.drop(columns=self.drop_cols).copy()
        self.df_5.reset_index(inplace=True)
        self.df_5.rename(columns={'sc-indx': 'sc_indx'}, inplace=True)
        print(f"Columns to be scaled except 'time': {self.df_5.columns}")
        for column in self.df_5.columns:
            if 'time' not in column:
                self.df_5[column] = (self.df_5[column]- self.df_5[column].min())/ (self.df_5[column].max() - self.df_5[column].min())
        
        self.omni_df['yrmndy_hr'] = pd.to_datetime(self.omni_df.yrmndy_hr)
        self.omni_df['time'] = (1000 - (pd.to_datetime('2050-01-01') - self.omni_df.yrmndy_hr).map(lambda x: 1000*(x.days + x.seconds/(24*60*60)))/self.delta_t).round(8)
        self.omni_df.drop(columns=['yrmndy_hr'], inplace=True)
        

    def make_ips_df(self):
        for x in self.file_name:
            indx_strt = self.df_test_1.shape[0]
            df_test_0 = pd.read_csv(self.ips_path + x, sep=r'\s+', skipinitialspace=True, skiprows=8, header=None, 
                                names=self.column_names[0:-1], usecols=[y for y in range(len(self.column_names[0:-1]))])
            df_test_0.index = df_test_0.index + indx_strt
            df_test_0['file'] = [x for i in range(df_test_0.shape[0])]
            self.df_test_1 = pd.concat([self.df_test_1, df_test_0])  
        del df_test_0
        print(f"Shape of IPS data: {self.df_test_1.shape}")

        def v_err(row):
            if type(row.v) == str:
                if row.v[-4:] == '-999':
                    row.v = int(row.v[0:-4])
                    row['sc-indx'] = row.er
                    row.er = -999
                else:
                    row.v = int(row.v)
            return row

        # Implementing v_err 
        self.df_test_1 = self.df_test_1.apply(v_err, axis=1)
        # print(f"{self.df_test_1.shape}")

        # Implementing one-hot encoding for error 'er' values of -999 in new coloumn 'er_1'

        self.df_test_1['er_1'] = self.df_test_1.er.map(lambda x: 1 if x==-999 else 0)

        # The 'yrmndy' is read as an integer, so converting it into a string and adding required zeros
        # for the first few years in the 2000s
        def yr_mod(col):
            x = str(col)
            if len(str(x))<6:
                x = ''.join([ '0' for j in range(6 - len(x)) ]) + str(x) 
            else:
                 x 
            return x
        # Implementing datetime stamp on 'yrmndy'
        self.df_test_1['yrmndy'] = pd.to_datetime(self.df_test_1.yrmndy.map(yr_mod), format='%y%m%d')
        print(f"{self.df_test_1.shape}")

        # Implementing conversion to datetime by adding yrmndy to it
        self.df_test_1['ut'] = self.df_test_1.yrmndy + pd.to_timedelta(self.df_test_1.ut, unit='h')
        return self.df_test_1

    def find_ranked_er(self, time, time_delta):
        """
        Parameters:
        --------------------------------------------------
        time: time in df_5.time format
        time_delta: time interval in df_5.time format


        Returns:
        -------------------------------------------------
        np.array of ranked list of df_5 indices according to least error i.e. df_5.er
        """
        df = self.df_5.loc[(self.df_5.time <= time) & (self.df_5.time > time - time_delta)]
        if len(df) > 0:
            return df.er.sort_values().index
        else:
            return np.array([])

    def fill_bracket(self, time_0, time_1, intervals):
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
        df = self.df_5.loc[(self.df_5.time <= time_0) & (self.df_5.time > time_1)].copy()


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
                obs_id = self.find_ranked_er(time_0 - i*time_delta, time_delta)     ## get list of indices with obs in the interval ranked acc. to error i.e. df_5.er
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
            bracket = bracket.reshape((-1,len(self.df_5.columns))) 
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

    def make_training_data(self, ips_data, omni_data, min_input_len=20):
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
            x_brckt = self.fill_bracket(time, time - self.delta_8, 32) # x_brckt has max len 32, it can be smaller
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
            if x_brckt_len == 32:
                time_0 = time*np.ones(32)
            else:
                time_0 = np.concatenate([time*np.ones(x_brckt_len), np.zeros(32 - x_brckt_len)])
                # print(j)
            # time_0 = time*np.ones(32)
            # x_brckt['time_trgt'] = pd.Series(time_0)
            x_brckt['time_trgt'] = time_0
            # Adding input column to indicate missing rows as 0
            # x_brckt['input'] = pd.Series(np.concatenate([np.ones(x_brckt_len), np.zeros(32 - x_brckt_len)]), dtype=float)
            x_brckt['input'] = np.concatenate([np.ones(x_brckt_len), np.zeros(32 - x_brckt_len)])
            x_brckt['time'] = x_brckt['time_trgt'] - x_brckt['time']


            # Test x_brckt['input'].sum()
            # print(x_brckt['input'].values.sum(), x_brckt_len, pd.Series(np.concatenate([np.ones(x_brckt_len), np.zeros(32 - x_brckt_len)]), dtype=float).values.sum())
            if x_brckt_len != x_brckt['input'].sum():
                print(x_brckt['input'].sum(), x_brckt_len, x_brckt.input.values)


            # Uncomment line below to return a list with X and y as pd.DataFrames
            # out_data.append([j, x_brckt, omni_data.iloc[i: i + 16] ]) 

            out_data.append([j] + list(x_brckt.values.reshape(-1)) + list(omni_data.iloc[i: i+16, 0].values.reshape(-1))) # choosing only one column from omni data

            # Keep track of x_brckt when len < 32
            if x_brckt_len < 32:
                missing.append([j, x_brckt_len])

            j = j + 1

        print(f"{k} Data points skipped due to lack of atleast {min_input_len} IPS data points in the past 8-days.")

        missing_df = pd.DataFrame(missing, columns=['id', 'missed'])
        if len(missing) > 0:
            print(missing_df.describe().to_string())
        print(f"{j} Data points made.")
        # return out_data, missing

        clmns_ips = list(ips_data.columns)
        clmns_ips.pop(0)
        clmns_ips.append('time_trgt')
        clmns_ips.append('input')
        print(clmns_ips, len(clmns_ips))
        clmns_input = []
        for i in range(32):
            for clmn in clmns_ips:
                clmns_input.append(f"X_{clmn}_{i}")
        # print(clmns_input)
        clmns_omni = list(omni_data.columns)
        clmns_target = []
        for i in range(16):
            for clmn in clmns_omni:
                if 'time' not in clmn:  # choosing only one column from omni data
                    clmns_target.append(f"y_{clmn}_{i}")
        clmns_data = ['idx'] + clmns_input + clmns_target

        out_df = pd.DataFrame(out_data, columns=clmns_data)

        return out_df, missing_df

    def make_final_data(self, ips_df, omni_df):
        # Making the final data 
        out_df, missing_df = self.make_training_data(ips_df, omni_df)
        
        print(f"scaling data's time columns")
        for column in out_df.columns:
            if "time" in column:
                out_df[column] = out_df[column]/1000
        return out_df, missing_df
        