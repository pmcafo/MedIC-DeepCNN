
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
import os
import pandas as pd
import json
from ast import literal_eval
import numpy as np
from plotnine import *
# Internal libraries/scripts
from ensmic.data_loading import IO_Inference
from ensmic.utils.metrics import compute_rawCM

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_data = "/home/mudomini/projects/ensmic/data"
path_results = "/home/mudomini/projects/ensmic/results"
phases = ["baseline", "augmenting", "stacking", "bagging"]
datasets = ["chmnist", "covid", "isic", "drd"]

#-----------------------------------------------------#
#                     Gather Data                     #
#-----------------------------------------------------#
cols = ["pd", "gt", "score", "dataset", "phase"]
df_results = pd.DataFrame(data=[], columns=cols)
# Iterate over phases
for phase in phases:
    # Iterate over each dataset
    for ds in datasets:
        # Load ground truth dictionary
        path_gt = os.path.join(path_data, ds + ".test.json")
        with open(path_gt, "r") as json_reader:
            gt_map = json.load(json_reader)

        # Get current path
        path_current = os.path.join(path_results, "phase_" + phase + "." + ds)
        # Identify best method
        if phase != "bagging":
            data = pd.read_csv(os.path.join(path_current, "evaluation",
                                            "results.test.averaged.csv"))
            if phase == "stacking":
                data.rename(columns={"ensembler": "method"}, inplace=True)
            else : data.rename(columns={"architecture": "method"}, inplace=True)
            data_f1 = data[data["metric"] == "F1"]
            best_dt = data_f1.iloc[data_f1["value"].argmax()]
            best_method = best_dt["method"]
        # For bagging: Identify best architecture & best method
        else:
            # iterate over all architectures
            best_score = 0
            for walk in os.listdir(path_current):
                if os.path.isfile(os.path.join(path_current, walk)) : continue
                # Load results
                data = pd.read_csv(os.path.join(path_current, walk, "evaluation",
                                                "results.test.averaged.csv"))
                # Obtain best
                data.rename(columns={"ensembler": "method"}, inplace=True)
                data_f1 = data[data["metric"] == "F1"]
                curr_dt = data_f1.iloc[data_f1["value"].argmax()]
                score = curr_dt["value"]
                if best_score > score : continue
                else:
                    best_score = score
                    best_architecture = walk
                    best_method = curr_dt["method"]
            # update path_current with best architecture
            path_current = os.path.join(path_current, best_architecture)

        # Create an Inference IO Interface
        if phase != "bagging" : path_inf = os.path.join(path_current,
                                            best_method, "inference.test.json")
        else : path_inf = os.path.join(path_current, "inference",
                            "inference." + best_method + ".pred.json")
        infIO = IO_Inference(None, path=path_inf)
        # Load predictions for samples
        inference = infIO.load_inference()
        # Load class names
        class_list = infIO.load_inference(index="legend")

        # Initialize lists for predictions and ground truth
        id = []