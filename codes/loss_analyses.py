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
loss_df["loss_arr"] = [
    np.load(
        loss_path + x,
         allow_pickle=True) for x in loss_df.loss_path]
# Find final loss values
loss_df["loss_arr_fin"] = [x[-1] for x in loss_df.loss_arr]


def get_config_corr(model_name: str):
	"""
	Get the best configuration (based on loss and y-scaled loss) and their corresponding
	correlations and mean squared errors (MSEs) for a given model name.

	This function filters a global ``loss_df`` DataFrame for entries corresponding to the
	specified model, identifies the configurations with minimum loss and minimum
	y-scaled loss, and then retrieves their associated correlation and MSE metrics from an
	external correlations file (``corrsM2e``).

	Parameters
	----------
	model_name : str
		Name of the model for which configurations and metrics should be retrieved.

	Returns
	-------
	tuple
		A 4-tuple containing:

		- **best_config** : str
			The configuration path corresponding to the lowest non-scaled loss.

		- **corr** : str
			Correlation and MSE table (as a string) for ``best_config``.

		- **best_y_config** : str
			The configuration path corresponding to the lowest y-scaled loss.

		- **corr_y** : str
			Correlation and MSE table (as a string) for ``best_y_config``.

	Notes
	-----
	- This function expects that a global DataFrame named ``loss_df`` exists
	  and contains at least the columns: ``loss_path`` and ``loss_arr_fin``.
	- Correlation values are read from ``../model_outputs/corrsM2e``, which is expected
	  to be a tab-separated file with predefined column names.
	- Printed output includes human-readable summaries of the selected configurations
	  and their losses before returning the results.
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
    print("\n", "Configuration with minimum loss")
    print(best_corr_str, "\n", "With minimum loss at")
    print(loss_model_df[loss_model_df.loss_path == best_config].loss_arr_fin.values)
    print("Related Correlators and MSEs")
    corr = corr_df[(corr_df.path == best_corr_str)].to_string()
    print(corr, "\n")
    # Find Corrs for lowest y scaled loss
    best_corr_y_str = best_y_config[:len(model_name)] + best_y_config[len(model_name) + 6:]
    print("Configuration with minimum y-scaled loss")
    print(best_corr_y_str, "\n", "With minimum loss at", "\n")
    print(loss_model_df[loss_model_df.loss_path == best_y_config].loss_arr_fin.values)
    print("Related Correlators and MSEs")
    corr_y = corr_df[(corr_df.path == best_corr_y_str)].to_string()
    print(corr_y, "\n")

    return best_corr_str, corr, best_corr_y_str, corr_y

def main():
    model_name = input("Enter model name: ").strip()
    get_config_corr(model_name)
    
if __name__=="__main__":
    main()