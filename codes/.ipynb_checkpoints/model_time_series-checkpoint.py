import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

class DatasetHist(Dataset):
    """
    Takes data from pd.DataFrame as produced by IPS_OMNI_make_data where 
    data is stored as a pd.DataFrame with columns=['idx', 'X...', 'y...']
    'X..': is of the form "X_{in_clmns}_{i}"
    in_clmns: is in  ['dist', 'hla', 'hlo', 'gla', 'glo', 'carr', 'v', 'er', 'sc-indx', 'time', 's_spts', 'time_trgt', 'input']
    i: is in range(32)
    Similarly 
    'y..': is of the form "y_{out_clmns}_{j}"
    out_clmns: is in [swSpeed_Smth, time]
    j: is in range(16)
    
    ----------------------------------------------------------------------------------------------------------------------------------------
    Params:
    file_path: path of pd.DataFrame produced by IPS_OMNI_make_data

    ----------------------------------------------------------------------------------------------------------------------------------------
    Returns:
    ID, X, y
    ID: index of data point
    X: input data np.array of shape (32, 13)
    y: target data np.array of shape (16,) 
    
    """
    def __init__(self, file_path:str, typ='train'):

        # Save data as pd.DataFrame with columns=['idx', 'X...', 'y...']
        # where 'X..': is of the form "X_{in_clmns}_{i}"
        # where in_clmns is in  ['dist', 'hla', 'hlo', 'gla', 'glo', 'carr', 'v', 'er', 'sc-indx', 'time', 's_spts', 'time_trgt', 'input']
        # and i is in range(32)
        # Similarly 'y..': is of the form "y_{out_clmns}_{j}"
        # where out_clmns is in [swSpeed_Smth]
        # and j is in range(16)
        
        self.data = pd.read_csv(file_path)
        self.ids = self.data.idx.values
        self.in_clmns = [x[2:-2] for x in list(self.data.columns)[1: 13 + 1]]
        print("Input features are", self.in_clmns)
        self.out_clmns = [x[2:-2] for x in list(self.data.columns)[1 + 13*32: 1 + 13*32 + 1]] # [1 + 13*32: 1 + 13*32 + 1] selects only thr wind-speed and not the time
        print("Target features are", self.out_clmns)
        self.X_clmns = [f"X_{in_clmn}_{i}" for i in range(32) for in_clmn in self.in_clmns]
        self.y_clmns = [f"y_{out_clmn}_{i}" for i in range(16) for out_clmn in self.out_clmns]
        # print("Input features", self.X_clmns, "\n", "Target features", self.y_clmns)
        print("begins:", self.ids[0],"ends:", self.ids[-1] )
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx:int):
        ID = self.ids[idx]
        X = self.data[self.X_clmns].loc[ID].values.reshape(32, -1).transpose() #------------------------# 13x32 np.array
        y = self.data[self.y_clmns].loc[ID].values.reshape(16, -1).transpose() #------------------------# 1x16  np.array, as time is not choosen in the target

        return ID, X, y
        