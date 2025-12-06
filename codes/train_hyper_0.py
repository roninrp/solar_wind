import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "..")
import numpy as np
from tqdm import tqdm
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
import codes.mnn_Utils as mnn
from codes.make_dataset import DatasetHist
from codes.data_utils import ips_omni_processor
import random

from codes.train_utils import cleanup

from termcolor import colored


# device = torch.device("cuda")
device = torch.device("xpu")

def train_hyperprams(
    model,
    optimizer,
    train_path:str, 
    val_path:str, 
    test_path:str,
    epochs:int,
    batch_size:int = 16,
    train_step:int=500,
    diff_alpha:int=0.0
):
    """
    Train a model using specified hyperparameters and evaluate performance on
    training, validation, and test datasets.

    This function orchestrates the full training loop, including dataset loading,
    dataloader construction, forward and backward passes, validation at each epoch,
    and final test evaluation using correlation and MSE metrics. The function
    returns training and validation losses per epoch, along with the best test-set
    statistics observed during training.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        Optimizer instance used for gradient updates.
    train_path : str
        Path to the training dataset file or directory compatible with
        `DatasetHist`.
    val_path : str
        Path to the validation dataset file or directory compatible with
        `DatasetHist`.
    test_path : str
        Path to the test dataset file or directory compatible with `DatasetHist`.
    epochs : int
        Number of training epochs.
    batch_size : int, optional
        Batch size used during training and validation. Defaults to ``16``.
    train_step: int =500,
        No.of. training batches to be trained upon.
    diff_alpha: int =0.0,
        alpha for difference in loss wrt time for back propagation.


    Returns
    -------
    train_Loss : numpy.ndarray of shape (epochs,)
        Scaled training loss per epoch (weighted MSE used for backpropagation).
    train_Loss_y : numpy.ndarray of shape (epochs,)
        Standard MSE training loss per epoch (unscaled).
    val_Loss : numpy.ndarray of shape (epochs,)
        Scaled validation loss per epoch.
    val_Loss_y : numpy.ndarray of shape (epochs,)
        Standard MSE validation loss per epoch.
    bestCorr : list of float
        Best Pearson correlation values for each monitored channel in the test set.
    bestMse : list of float
        Best MSE values (scaled by 800) for each monitored channel in the test set.
    bestEpoch : int
        Epoch index at which the best correlation performance was observed.

    Notes
    -----
    - Training loss used for backpropagation is scaled by the target values and a
      factor of 100. The unscaled MSE is stored separately.
    - Best test-set correlation and MSE are computed on four reference channels
      defined by indices ``[9, 11, 13, 15]``.
    - The scheduler used for learning rate adjustments is expected to be defined
      externally in the calling scope.
    - Global variables such as ``device``, ``lr``, ``weight_decay``, ``dropout``,
      ``gamma``, and ``scheduler`` must be defined outside this function.

    """

    # Load Datasets
    train_ds = DatasetHist(train_path)
    val_ds = DatasetHist(val_path)
    test_ds = DatasetHist(test_path)

    # Load DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    model = model.to(device)

    # Compute the difference in derivative wrt time between y_out and y
    def diff_loss(y_out, y):
        y_len = y.shape[0]
        loss_diff = torch.empty((y_len - 1, 1, 16), dtype=torch.float32, device= device)
        loss_diff = F.mse_loss((y[1:, :, :] - y[:-1, :, :]), (y_out[1:, :, :] - y_out[:-1, :, :]))
        return loss_diff
        
    # Compute the 1 - corr**2 between y_out and y
    def corr_loss(y_out, y):
        y_len = y.shape[0]
        a = y_out - torch.mean(y_out, dim=0, keepdim=True)
        b = y - torch.mean(y, dim=0, keepdim=True)
        cov_ab = torch.sum(a * b, dim=0)
        loss_corr = 1 - cov_ab**2/(torch.sum(a**2, dim=0) * torch.sum(b**2, dim=0) + 1e-8)
        return loss_corr.mean()

    # Run model on data, used both on train and val
    # Define Backprop loss here
    def doStep(data):
        x = data[1].to(torch.float32)
        x = x.to(device)
        # print("x", x)
        y = data[2].to(torch.float32)
        y = y.to(device)
        y_out = model(x)
        # print("Before backward:", torch.isnan(y_out).any())

        loss = F.mse_loss(y_out, y, reduction='none')
        loss = loss * y * 100.0 
        loss_diff = diff_loss(y_out, y)                   # derviative of loss wrt time i.e. loss difference in a batch
        loss = loss.mean()                            # Loss used for Backprop, y scaled
        y_loss = F.mse_loss(y_out, y)                 # The actual loss: mse

        return loss, y_loss, loss_diff

    # Training for an epoch
    def train_epoch(train_dl, epoch, train_step:int=500):
        train_dl_len = len(train_dl)
        model.eval()
        running_loss = 0.0
        running_loss_y = 0.0

        loop = tqdm(enumerate(train_dl), total=train_step, leave=False)
        for i, data in loop:
            # zero the parameter gradients
            optimizer.zero_grad()
            loss, y_loss, loss_diff = doStep(data)
            # print(i, "epoch:", epoch, ";", "loss", loss.item())
            del data

            # print(loss.dtype)
            loss_bkprp = loss + diff_alpha * loss_diff          # adding final loss for back propagation
            loss_bkprp.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_y += y_loss.item()

            # update progress bar:
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=y_loss.item(), dA=diff_alpha)
            if i == train_step:
                break

        # Epoch averages
        running_loss = running_loss / train_step
        running_loss_y = running_loss_y / train_step
        print("")
        # print(lr, weight_decay, dropout, batch_size, gamma)
        print("Epoch: ", epoch, "Error: ", np.sqrt(running_loss_y) * 800.0)
        return running_loss, running_loss_y

    # Validation for an epoch
    def validate_epoch(val_dl, epoch):
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_loss_y = 0.0
            val_dl_len = len(val_dl)

            loop = tqdm(enumerate(val_dl), total=val_dl_len, leave=False)
            for i, data in loop:
                loss, y_loss, _ = doStep(data)
                # print("val loss", loss)
                del data

                running_loss += loss.item()
                running_loss_y += y_loss.item()

                loop.set_description(f"Validate [{i}/{val_dl_len}]")
                loop.set_postfix(loss=loss.item())

            # Epoch average losses 
            running_loss = running_loss / val_dl_len
            running_loss_y = running_loss_y / val_dl_len
            print("Epoch: ", epoch, "Error: ", np.sqrt(running_loss_y) * 800.0)
        return running_loss, running_loss_y

    # Test evlaution for an epoch
    def test_epoch(test_dl, epoch, bestCorr, bestMse, bestEpoch):
        model.eval()
        with torch.no_grad():
            for batch in test_dl:
                testBatch = batch
                break
            x = testBatch[1].to(torch.float32)
            x = x.to(device)
            # y = testBatch[2]
            opY = model(x).detach().cpu().numpy()

        refY = testBatch[2].numpy()
        # print("refY.shape", refY.shape)

        corrVals = [-1.0, -1.0, -1.0, -1.0]
        mseVals = [300.0, 300.0, 300.0, 300.0]
        refIds = [9, 11, 13, 15]
        for i in range(4):
            thisRef = refY[:, :, refIds[i]].flatten()
            thisOp = opY[:, :, refIds[i]].flatten()
            corrVals[i] = pearsonr(thisRef, thisOp)[0]
            mseVals[i] = np.mean(np.sqrt((thisOp - thisRef)**2) * 800.0)
        if corrVals[-1] > bestCorr[-1]:
            bestCorr = corrVals
            # torch.save(
            #     model.state_dict(), resultsPath + 'models/%s_%s' %
            #     (mString, paraString))
            bestMse = mseVals
            bestEpoch = epoch
        print(
            "Evaluate:  ",
            " epoch:",
            epoch,
            ", bestCorr:",
            bestCorr,
            ", bestEpoch:",
            bestEpoch)
        return bestCorr, bestMse, bestEpoch


    # Initialize loss arrays
    train_Loss_y = np.empty(epochs, dtype=np.float32)
    train_Loss = np.empty(epochs, dtype=np.float32)
    val_Loss_y = np.empty(epochs, dtype=np.float32)
    val_Loss = np.empty(epochs, dtype=np.float32)

    # Initialize Bests
    bestCorr = [-1.0, -1.0, -1.0, -1.0]
    print('bestCorr is defined')
    bestMse = [999.0, 999.0, 999.0, 999.0]
    bestEpoch = epochs
    
    for epoch in range(1, epochs + 1):
        print(
            f"Training epoch {epoch} =================================================================================")
        tlosses = train_epoch(train_dl=train_dl, epoch=epoch, train_step=train_step)
        print(f"Validating epoch {epoch}")
        vlosses = validate_epoch(val_dl=val_dl, epoch= epoch)
        scheduler.step()

        # Scaled losses: y x 100 x mse
        train_Loss[epoch - 1] = tlosses[0]
        val_Loss[epoch - 1] = vlosses[0]

        # Usual losses: mse
        train_Loss_y[epoch - 1] = tlosses[1]
        val_Loss_y[epoch - 1] = vlosses[1]

        if epoch > 1:
            print(f"Evaluating epoch {epoch}")
            bestCorr, bestMse, bestEpoch = test_epoch(test_dl, epoch, bestCorr, bestMse, bestEpoch)

    return train_Loss, train_Loss_y, val_Loss, val_Loss_y, bestCorr, bestMse, bestEpoch


if __name__ == "__main__":
    # define model
    mString = 'base_v0'
    # Define learning parameters ---------------------------------------------
    lrs = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6][1:2]  # learning rate
    weight_decays = [1.0e-4, 1.0e-5][:1]  # weight decay regularization
    dropouts = [0.3, 0.2, 0.4][1:2]  # dropouts
    batch_sizes = [16, 32][-1:]      # batchsize
    # ,0.80] #learning rate scheduler, reduces the learning rate as training gets cloes to the minima
    gammas = [0.95, 0.90, 0.85][-1:]
    diff_alphas = [0.0, 1.0, 5.0, 10.0, 50.0][:1]
    kwargs = {'num_workers': 4, 'pin_memory': True}
    torch.manual_seed(42)
    resultsPath = "../model_outputs_corrected/local_run/"
    # Inititialize optimizer and scheduler
    
    epochs = 5
    startT = time.time()
    
    for lr in lrs:
        for weight_decay in weight_decays:
            for dropout in dropouts:
                for batch_size in batch_sizes:
                    for gamma in gammas:
                        for diff_alpha in diff_alphas:
                            # Cleanup memory
                            try:
                                cleanup(model, optimizer)
                            except NameError:
                                pass
                            # load model, optimizer and scheduler
                            model = mnn.model_base(xCh=13, LEN=32, emDim=128, dropout=0.2, device=device)
                            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
                            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        
                            # get values:
                            train_Loss, train_Loss_y, val_Loss, val_Loss_y, bestCorr, bestMse, bestEpoch = train_hyperprams(
                                model=model,
                                optimizer=optimizer,
                                train_path="../data/data_generated/train_corrected/train_ips_omni_df.csv",
                                val_path="../data/data_generated/val_corrected/val_ips_omni_df.csv",
                                test_path="../data/data_generated/test_corrected/test_ips_omni_df.csv",
                                epochs=epochs,
                                batch_size=batch_size,
                                train_step=100,
                                diff_alpha=diff_alpha
                            )
        
                            # String to store model training parameters
                            paraString = 'lr%0.2e_wd%0.2e_drp%0.2f_bS%d_E%d_sR%0.2f_dA%0.2e' % (
                                lr, weight_decay, dropout, batch_size, epochs, gamma, diff_alpha)
                            
                            # Store losses
                            train_Loss.dump(resultsPath + 'losses/%s_train_%s' % (mString, paraString))
                            val_Loss.dump(resultsPath + 'losses/%s_val_%s' % (mString, paraString))
                            train_Loss_y.dump(resultsPath + 'losses/%s_train_y_%s' % (mString, paraString))
                            val_Loss_y.dump(resultsPath + 'losses/%s_val_y_%s' % (mString, paraString))
                            # Store correlations and statistics
                            with open(resultsPath + 'corrsM2e', 'a') as fl:
                                # fl.write('%s_%s\t%0.6f\t%0.6f\t%d\n'%(mString,paraString,bestCorr,bestMse,bestEpoch))
                                fl.write('%s_%s' % (mString, paraString))
                                for i in range(4):
                                    fl.write('\t%0.6f\t%0.6f' % (bestCorr[i], bestMse[i]))
                                fl.write('\t%d\n' % bestEpoch)
                            print("lr:",lr, "wd:",weight_decay, "drpt:",dropout, "bs:",batch_size, "g:",gamma, "time:",time.time() - startT)
