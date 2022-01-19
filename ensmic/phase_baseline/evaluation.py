
#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import argparse
import os
import json
import pandas
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference, architecture_list, architecture_params
from ensmic.utils.metrics import compute_metrics, compute_rawCM
from ensmic.utils.categorical_averaging import macro_averaging, macro_average_roc
# Experimental
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'chmnist', 'drd']",
                    required=True, type=str, dest="seed")
args = parser.parse_args()

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Initialize configuration dictionary
config = {}
# Path to data directory
config["path_data"] = "data"
# Path to result directory
config["path_results"] = "results"
# Seed (if training multiple runs)
config["seed"] = args.seed

#-----------------------------------------------------#
#               Function: Preprocessing               #
#-----------------------------------------------------#
def preprocessing(architecture, dataset, config):
    # Load ground truth dictionary
    path_gt = os.path.join(config["path_data"], config["seed"] + \
                           "." + dataset + ".json")
    with open(path_gt, "r") as json_reader:
        gt_map = json.load(json_reader)
    # Get result subdirectory for current architecture
    path_arch = os.path.join(config["path_results"], "phase_baseline" + "." + \
                             config["seed"], architecture)
    # Create an Inference IO Interface
    path_inf = os.path.join(path_arch, "inference" + "." + dataset + ".json")
    infIO = IO_Inference(None, path=path_inf)
    # Load predictions for samples
    inference = infIO.load_inference()
    # Load class names
    config["class_list"] = infIO.load_inference(index="legend")

    # Initialize lists for predictions and ground truth
    id = []
    gt = []
    pd_class = []
    pd_prob = []
    # Iterate over all samples of the testing set
    sample_list = inference.keys()
    for sample in sample_list:
        # Obtain ground truth and predictions
        id.append(sample)
        gt.append(np.argmax(gt_map[sample]))
        prediction = inference[sample]
        pd_class.append(np.argmax(prediction))
        pd_prob.append(prediction)
    # Return parsed information
    return id, gt, pd_class, pd_prob

#-----------------------------------------------------#
#          Function: Results Parsing & Backup         #
#-----------------------------------------------------#
def parse_results(metrics, architecture, dataset, config):
    # Parse metrics to Pandas dataframe
    results = pandas.DataFrame.from_dict(metrics)
    results = results.transpose()
    results.columns = config["class_list"]
    # Backup to disk
    path_arch = os.path.join(config["path_results"], "phase_baseline" + "." + \
                             config["seed"], architecture)
    path_res = os.path.join(path_arch, "metrics." + dataset + ".csv")
    results.to_csv(path_res, index=True, index_label="metric")
    # Return dataframe
    return results

def collect_results(result_set, architectures, dataset, path_eval, config):
    # Initialize result dataframe
    cols = ["architecture", "class", "metric", "value"]
    df_results = pandas.DataFrame(data=[], dtype=np.float64, columns=cols)
    # Iterate over each architecture results
    for i in range(0, len(architectures)):
        arch_type = architectures[i]
        arch_df = result_set[i].copy()
        arch_df.drop(index="TP-TN-FP-FN", inplace=True)
        arch_df.drop(index="ROC_FPR", inplace=True)
        arch_df.drop(index="ROC_TPR", inplace=True)
        arch_df = arch_df.astype(float)
        # Parse architecture result dataframe into desired shape
        arch_df = arch_df.reset_index()
        arch_df.rename(columns={"index":"metric"}, inplace=True)
        arch_df["architecture"] = arch_type
        arch_df = arch_df.melt(id_vars=["architecture", "metric"],
                               value_vars=config["class_list"],
                               var_name="class",
                               value_name="value")
        # Reorder columns
        arch_df = arch_df[cols]
        # Merge to global result dataframe
        df_results = df_results.append(arch_df, ignore_index=True)
    # Backup merged results to disk
    path_res = os.path.join(path_eval, "results." + dataset + ".collection.csv")
    df_results.to_csv(path_res, index=False)
    # Return merged results
    return df_results

#-----------------------------------------------------#
#        Compute and Store raw Confusion Matrix       #
#-----------------------------------------------------#
def calc_confusion_matrix(gt, pd, architecture, dataset, config):
    # Compute confusion matrix
    rawcm_np = compute_rawCM(gt, pd, config["class_list"])
    rawcm = pandas.DataFrame(rawcm_np)
    # Tidy dataframe
    rawcm.index = config["class_list"]
    rawcm.columns = config["class_list"]
    # Backup to disk
    path_arch = os.path.join(config["path_results"], "phase_baseline" + "." + \
                             config["seed"], architecture)
    path_res = os.path.join(path_arch, "confusion_matrix." + dataset + ".csv")
    rawcm.to_csv(path_res, index=True, index_label="metric")
    # Return results
    return rawcm
