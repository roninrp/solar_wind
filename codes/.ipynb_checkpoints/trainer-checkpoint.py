from codes.make_dataset import DatasetHist
import codes.mnn_Utils as mnn
from tqdm import tqdm
from termcolor import colored
import random
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
from math import sqrt
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import os
import torch.nn as nn
import torch
import pickle
from scipy.stats import pearsonr
import datetime as dt
import pandas as pd
import numpy as np
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "..")


runCount = 0
# device = torch.device("cuda")
device = torch.device("xpu")
epochs = 10
train_steps = 1000

# Load path for output----------------------------------------------------
resultsPath = "../model_outputs/"
mString = 'base_test_en_1by30'

# Load paths for train, val and test -------------------------------------
train_path = "../data/data_generated/train/train_ips_omni_df.csv"
val_path = "../data/data_generated/val/val_ips_omni_df.csv"
test_path = "../data/data_generated/test/test_ips_omni_df.csv"

# Make Datasets
train_set = DatasetHist(train_path)
val_set = DatasetHist(val_path)
test_set = DatasetHist(test_path)

# Define training fucntion: trains, validates, tests and stores output


def start_training(train_set, val_set, test_set, train_steps, epochs):
    """
    Parameters:
    -----------
    train_set: Training Dataset made via DatasetHist
    val_set: Val Dataset made via DatasetHist
    test_set: Test Dataset made via DatasetHist

    train_steps: int number of batches to be trained
    epochs: number of epochs to be trained
    """
    # Get data loaded
    train_load = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    val_load = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **kwargs)
    test_load = DataLoader(
        test_set,
        batch_size=len(test_set),
        shuffle=False,
        **kwargs)

    # Load model
    model = mnn.model_base(
        xCh=13,
        LEN=32,
        emDim=128,
        dropout=dropout,
        device=device)
    # Use GPU, only when available
    model = model.to(device)

    # Inititialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Initialize loss arrays
    train_Loss_y = np.empty(epochs, dtype=np.float32)
    train_Loss_diff = np.empty(epochs, dtype=np.float32)
    train_Loss = np.empty(epochs, dtype=np.float32)
    val_Loss_y = np.empty(epochs, dtype=np.float32)
    val_Loss_diff = np.empty(epochs, dtype=np.float32)
    val_Loss = np.empty(epochs, dtype=np.float32)

    # Initialize Bests
    bestCorr = [-1.0, -1.0, -1.0, -1.0]
    print('bestCorr is defined')
    bestMse = [999.0, 999.0, 999.0, 999.0]
    bestEpoch = epochs

    # String to store model training parameters
    paraString = 'lr%0.2e_wd%0.2e_drp%0.2f_bS%d_E%d_sR%0.2f' % (
        lr, weight_decay, dropout, batch_size, epochs, gamma)

    def train(data_load, epoch, run_count=run_count, train_steps=train_steps):
        model.train()
        running_loss = 0.0
        running_loss_y = 0.0

        loop = tqdm(enumerate(data_load), total=len(data_load), leave=False)
        for i, data in loop:
            # zero the parameter gradients
            optimizer.zero_grad()
            loss, y_loss = doStep(data)
            # print(i, "epoch:", epoch, ";", "loss", loss.item())
            del data

            # print(loss.dtype)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_y += y_loss.item()

            # update progress bar:
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            if i == train_steps:
                break

        # print statistics
        running_loss = running_loss / train_steps
        running_loss_y = running_loss_y / train_steps
        print("")
        print(lr, weight_decay, dropout, batch_size, gamma)
        print('Train:  Run %d [Epoch %d] loss: %.4e' %
              (run_count, epoch, running_loss_y * 800.0))
        return running_loss, running_loss_y

    def doStep(data):
        x = data[1].to(torch.float32)
        x = x.to(device)
        # print("x", x)
        y = data[2].to(torch.float32)
        y = y.to(device)
        y_out = model(x)
        # print("Before backward:", torch.isnan(y_out).any())

        loss = F.mse_loss(y_out, y, reduction='none')
        loss = loss * y * 100.0  # Why this scaling
        loss = loss.mean()
        y_loss = F.mse_loss(y_out, y)

        return loss, y_loss

    def validate(val_steps, epoch, run_count):
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_loss_y = 0.0

            loop = tqdm(enumerate(val_load), total=len(val_load), leave=False)
            for i, data in loop:
                loss, y_loss = doStep(data)
                print("val loss", loss)
                del data

                running_loss += loss.item()
                running_loss_y += y_loss.item()

                loop.set_description(f"Validate [{i}/{len(val_load)}]")
                loop.set_postfix(loss=loss.item())
                if i == val_steps:
                    break

            # print statistics
            running_loss = running_loss / val_steps
            running_loss_y = running_loss_y / val_steps
            print(
                'Validate:  Run %d [Epoch %d] val loss: %.4e' %
                (run_count, epoch, running_loss_y * 800.0))
        return running_loss, running_loss_y

    def evalModel(bestCorr, bestMse, bestEpoch):
        model.eval()
        with torch.no_grad():
            for batch in test_load:
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
            torch.save(
                model.state_dict(), resultsPath + 'models/%s_%s' %
                (mString, paraString))
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

    for epoch in range(1, epochs + 1):
        print(
            f"Training epoch {epoch} =================================================================================")
        tlosses = train(train_load, epoch, run_count, train_steps)
        print(f"Validating epoch {epoch}")
        vlosses = validate(len(val_set) // batch_size, epoch, run_count)
        scheduler.step()

        train_Loss[epoch - 1] = tlosses[0]
        val_Loss[epoch - 1] = vlosses[0]

        train_Loss_y[epoch - 1] = tlosses[1]
        val_Loss_y[epoch - 1] = vlosses[1]

        if epoch > 1:
            print(f"Evaluating epoch {epoch}")
            bestCorr, bestMse, bestEpoch = evalModel(
                bestCorr, bestMse, bestEpoch)

    train_Loss.dump(resultsPath + 'losses/%s_train_%s' % (mString, paraString))
    val_Loss.dump(resultsPath + 'losses/%s_val_%s' % (mString, paraString))
    train_Loss_y.dump(
        resultsPath + 'losses/%s_train_y_%s' %
        (mString, paraString))
    val_Loss_y.dump(resultsPath + 'losses/%s_val_y_%s' % (mString, paraString))
    with open(resultsPath + 'corrsM2e', 'a') as fl:
        # fl.write('%s_%s\t%0.6f\t%0.6f\t%d\n'%(mString,paraString,bestCorr,bestMse,bestEpoch))
        fl.write('%s_%s' % (mString, paraString))
        for i in range(4):
            fl.write('\t%0.6f\t%0.6f' % (bestCorr[i], bestMse[i]))
        fl.write('\t%d\n' % bestEpoch)
    print(lr, time.time() - startT)
    return


# Define learning parameters ---------------------------------------------
lrs = [1.0e-3, 1.0e-2]  # learning rate
weight_decays = [1.0e-4, 1.0e-3]  # weight decay regularization
dropouts = [0.3, 0.2, 0.4]  # dropouts
batch_sizes = [32, 16, 64]      # batchsize
# ,0.80] #learning rate scheduler, reduces the learning rate as training gets cloes to the minima
gammas = [0.95, 0.90, 0.85]
kwargs = {'num_workers': 4, 'pin_memory': True}
torch.manual_seed(1)

# Define run count
run_count = 0

if __name__ == "__main__":
    startT = time.time()
    pbar = tqdm(
        total=len(lrs) *
        len(weight_decays) *
        len(dropouts) *
        len(batch_sizes) *
        len(gammas),
        leave=False)
    for lr in lrs:
        for weight_decay in weight_decays:
            for dropout in dropouts:
                for batch_size in batch_sizes:
                    for gamma in gammas:
                        start_training(
                            train_set, val_set, test_set, train_steps, epochs)
                        pbar.update(1)
                        pbar.set_description(f" Hyper-parameters")
                        print(lr, weight_decay, dropout, batch_size, gamma)
                        run_count += 1
    pbar.close()
    print(time.time() - startT)
