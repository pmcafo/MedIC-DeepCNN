
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
import time
import json
import pandas as pd
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.ensemble import ensembler_dict, ensembler

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Analysis of COVID-19 Classification via Ensemble Learning")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'chmnist', 'drd']",
                    required=True, type=str, dest="seed")
parser.add_argument("-g", "--gpu", help="GPU ID selection for multi cluster",
                    required=False, type=int, dest="gpu", default=0)
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
# List of ensemble learning techniques
config["ensembler_list"] = ensembler

#-----------------------------------------------------#
#                     Run Training                    #
#-----------------------------------------------------#
def run_training(ds_x, ds_y, ensembler, path_elm, config):
    # Obtain initialization variables for Ensemble Learning model
    n_classes = len(config["class_list"])
    # Create Ensemble Learning model
    model = ensembler_dict[ensembler](n_classes=n_classes)
    # Fit model on data
    model.training(ds_x.copy(), ds_y.copy())
    # Dump fitted model to disk
    path_model = os.path.join(path_elm, "model.pkl")
    model.dump(path_model)
    # Return fitted model and esembler path
    return model

#-----------------------------------------------------#
#                    Run Inference                    #
#-----------------------------------------------------#
def run_inference(test_x, model, path_elm, config):
    # Compute predictions via Ensemble Learning method
    predictions = model.prediction(test_x.copy())

    # Create an Inference IO Interface
    path_inf = os.path.join(path_elm, "inference" + "." + "test" + ".json")
    infIO = IO_Inference(config["class_list"], path=path_inf)
    # Store prediction for each sample
    samples = test_x.index.values.tolist()
    infIO.store_inference(samples, predictions)

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Identify phase results directory
path_phase = os.path.join(config["path_results"],
                          "phase_stacking" + "." + str(config["seed"]))
# Load dataset for training
train_x = pd.read_csv(os.path.join(path_phase, "phase_baseline.inference." + \
                                   "val-ensemble." + "set_x" + ".csv"),
                      header=0, index_col="index")
train_y = pd.read_csv(os.path.join(path_phase, "phase_baseline.inference." + \
                                   "val-ensemble." + "set_y" + ".csv"),
                      header=0, index_col="index")
# Load dataset for testing
test_x = pd.read_csv(os.path.join(path_phase, "phase_baseline.inference." + \
                                  "test." + "set_x" + ".csv"),
                     header=0, index_col="index")

# Load class list
path_gt = os.path.join(config["path_data"], config["seed"] + \
                       "." + "val-ensemble" + ".json")
with open(path_gt, "r") as json_reader:
    gt_map = json.load(json_reader)
config["class_list"] = gt_map["legend"]

# Initialize cache memory to store meta information
timer_cache = {}

# Run Training and Inference for all ensemble learning techniques
for ensembler in config["ensembler_list"]:
    print("Start running Ensembler:", ensembler)
    try:
        # Get path to Ensembler subdirectory
        path_elm = os.path.join(path_phase, ensembler)
        # Run Pipeline
        timer_start = time.time()
        model = run_training(train_x, train_y, ensembler, path_elm, config)
        run_inference(test_x, model, path_elm, config)
        timer_end = time.time()
        # Store execution time in cache
        timer_time = timer_end - timer_start
        timer_cache[ensembler] = timer_time
        print("Finished running Ensembler:", ensembler, timer_time)
    except Exception as e:
        print(ensembler, "-", "An exception occurred:", str(e))

# Store time measurements as JSON to disk
path_time = os.path.join(config["path_results"], "phase_stacking" + "." + \
                         config["seed"], "time_measurements.json")
with open(path_time, "w") as file:
    json.dump(timer_cache, file, indent=2)