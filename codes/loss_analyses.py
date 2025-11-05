import pandas as pd
import numpy as np
import os
import pickle

pd.set_option('display.max_colwidth', None)

# Path of losses folder
loss_path = "../model_outputs/losses/"

# Get list of all .pkl files storing losses
loss_path_lists = os.listdir(loss_path)

# Make Data frame
loss_df = pd.DataFrame({"loss_path": loss_path_lists})
# Load losses for all files
loss_df["loss_arr"] = [np.load(loss_path + x, allow_pickle=True) for x in loss_df.loss_path]
# Find final loss values
loss_df["loss_arr_fin"] = [x[-1] for x in loss_df.loss_arr]




def get_config_corr(model_name: str):
    """
    Returns the best configuration and corresponding correlation and mse for the given model name.

    Prameter:
    ---------
    
    model_name = str name of the model

    Returns:
    --------
    
    Tuple: (best_config, corr, best_y_config, corr_y)
    
    best_config = Best config with lowest loss
    corr = corresponding correlations and mse
    best_y_config = Best config with lowest y-scaled loss
    corr_y = corr = corresponding correlations and mse
    """
    # Make dataframe for the given mode
    loss_model_df = loss_df[(loss_df.loss_path.apply(lambda x: x[:len(model_name) + 6]) == model_name + "_val_y") | (loss_df.loss_path.apply(lambda x: x[:len(model_name) + 6]) == model_name + "_val_l") ]. copy()
    # Create index for val loss measured is scaled with y or not
    loss_model_df["y_or_not"] = [0 if x[:len(model_name) + 6] == model_name + "_val_l" else 1 for x in loss_model_df.loss_path]
    # Find best config with lowest y scaled loss
    best_y_config = loss_model_df[(loss_model_df.y_or_not == 1) & (loss_model_df[(loss_model_df.y_or_not == 1)].loss_arr_fin == loss_model_df[(loss_model_df.y_or_not == 1)].loss_arr_fin.min())].loss_path.iloc[0]
    # Find best config for lowest loss
    best_config = loss_model_df[(loss_model_df.y_or_not == 0) & (loss_model_df[(loss_model_df.y_or_not == 0)].loss_arr_fin == loss_model_df[(loss_model_df.y_or_not == 0)].loss_arr_fin.min())].loss_path.iloc[0]
    
    # Define Corr dataframe
    corr_df = pd.read_csv("../model_outputs/corrsM2e", sep="\t", names=["path", "corr_9", "mse_9", "corr_11", "mse_11", "corr_13", "mse_13", "corr_15", "mse_15", "best_epoch"])
    # Find Corrs for lowest loss
    best_corr_str = best_config[:len(model_name)] + best_config[len(model_name) + 4:] 
    print(best_corr_str)
    corr = corr_df[(corr_df.path == best_corr_str)].to_string()
    print(corr)
    # Find Corrs for lowest y scaled loss
    best_corr_y_str = best_y_config[:len(model_name)] + best_y_config[len(model_name) + 6:]
    print(best_corr_y_str)
    corr_y = corr_df[(corr_df.path == best_corr_y_str)].to_string()
    print(corr_y)

    return best_corr_str, corr, best_corr_y_str, corr_y

def main():
    model_name = input("Enter model name: ").strip()
    get_config_corr(model_name)
    
if __name__=="__main__":
    main()