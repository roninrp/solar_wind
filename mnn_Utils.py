import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
import pickle
import torch
import torch.nn as nn
import os
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from math import sqrt
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torchvision import models
device = torch.device("cuda") 

##############-------------Positional_Encoder---------------------------####################################################################
##############---------------posEncoder____________3--------------------####################################################################
class posEncoder(nn.Module):
    ## This will take the X and sum them with PE, we will perform the layernorm and add
    def __init__(self, emDim=128, ipLen=32, device=device):
        super(posEncoder,self).__init__()
        
        posEm = np.empty((emDim,ipLen),dtype='float32')
        for i in range(0,emDim//2):
            for j in range(ipLen):
                posEm[2*i,j] = np.cos(float(j)/((2.0*ipLen)**(2.0*i/emDim)))
                posEm[2*i+1,j] = np.sin(float(j)/((2.0*ipLen)**(2.0*i/emDim)))
        posEm = np.expand_dims(posEm, axis=0)
        self.posEm = torch.from_numpy(posEm).to(device)
        
        self.xLn = nn.LayerNorm((emDim,ipLen))
        self.peLn = nn.LayerNorm((emDim,ipLen))

    def forward(self,X,T):
        return self.xLn(X),  self.peLn(self.posEm)


##################---------------Encoder-------------------------###########################################################################
##################-----------conv1Dencoder____________1----------###########################################################################
class conv1Dencoder(nn.Module):
    def __init__(self, xCh=13, LEN=32, emDim=128, device = device):
        super(conv1Dencoder,self).__init__()
        self.xProj = nn.Conv1d(xCh,emDim,1,1)                           #---------------------# 13 features of input X (batch, features, seq_length) are mapped to 
                                                                                              # (input)channels xCh=13
        self.posEm = posEncoder(emDim=emDim, ipLen=LEN, device = device) #------------------# Postional_Encoding: ipLen is length of input X (i.e. seq length)

        # self.conv1d takes a matrix of dimensions  channels x images
        # here image starts off as of LEN=32 referring to the 32 equally spaced intervals spanning 8-days
        # and channels starts off as xCh or tCh = 7 referring to the 7 different features which then transforms to 128 after posEncoder

        self.conv1 = nn.Conv1d(emDim,256,3,1,padding='same') # ---------- # channels: 128->256, image:32, final: 2x256
        self.pool1 = nn.MaxPool1d(2)                         # ---------- # image: 32-> 32/2, final: channelx16
        self.proj1 = nn.Conv1d(emDim,256,1,2)                # ---------- # channels: emDim->256, image: image/2, final: (image/2)x256
        
        self.conv2 = nn.Conv1d(256,512,3,1,padding='same')
        self.pool2 = nn.MaxPool1d(2)
        self.proj2 = nn.Conv1d(256,512,1,2)
        
        self.conv3 = nn.Conv1d(512,1024,3,1,padding='same')
        self.pool3 = nn.MaxPool1d(2)
        self.proj3 = nn.Conv1d(512,1024,1,2)

    def forward(self,X,T):
        x1 = self.posEm(self.xProj(X))

        x2  = torch.relu( self.pool1( self.conv1(x1) ) )
        x2  = x2 + self.proj1(x1)
        
        x3  = torch.relu( self.pool2( self.conv2(x2) ) )
        x3  = x3 + self.proj2(x2)
        
        x4  = torch.relu( self.pool3( self.conv3(x3) ) )
        x4  = x4 + self.proj3(x3) # ------------- 4x1024

        return torch.flatten(x4,1) # ------------- (N, 4096)             # needs to return (N, 2, 2048) where one of the features would morph into the target time column.

###################-----------Model_block-----------------###############################################################################
#########################################################################################################################################
# Needs to change to take input of the shape (N, 2, 2048) instead of (N, 1, 4096) so as to give an output of the shape (N, 2, 16) instead of (N, 16)

class model_block(nn.Module):
    def __init__(self, xCh = 13, LEN=32, emDim=128, dropout=0.3, device=device):
        super(model_block, self).__init__()
        self.enc1 = conv1Dencoder(xCh=xCh,  LEN=LEN, emDim=emDim, device = device)
        self.ln1 = nn.LayerNorm(4096)
        self.fc1 = nn.Linear(4096,1024)
        self.pr1 = nn.Linear(4096,256)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,64)
        self.output = nn.Linear(64,16)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        xenEm = self.ln1( self.enc1(X) )                     #-------------# 4096
        x1 = torch.sigmoid( self.fc1(xenEm) )                #-------------# 1024
        x2 = torch.sigmoid( self.fc2(x1) ) + self.pr1(xenEm) #-------------# 256
        x2 = self.dropout(x2)                                #-------------# 256
        x3 = self.dropout( torch.sigmoid( self.fc3(x2) ) )   #-------------# 64
        return torch.sigmoid( self.output(x3) )              #-------------# 16
