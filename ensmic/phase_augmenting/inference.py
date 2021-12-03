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
# AUCMEDI libraries
from aucmedi import DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.neural_network.architectures import supported_standardize_mode, \
                                                 architecture_dict
from aucmedi.ensembler import predict_augmenting
# ENSMIC libraries
from ensmic.data_loading import IO_Inference, load_sampling, architecture_list

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

# Preprocessor Configurations
config["threads"] = 16
config["batch_size"] = 32
config["batch_queue_size"] = 16
# Neural Network Configurations
config["workers"] = 16

# Adjust GPU configuration
config["gpu_id"] = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

#-----------------------------------------------------#
#                   AUCMEDI Pipeline                  #
#-----------------------------------------------------#
def run_aucmedi(samples, dataset, architecture, config, best_model=True):
    # Define Subfunctions
    sf_list = [Padding(mode="square")]
    # Set activation output to softmax for multi-class classification
    activation_output = "softmax"

    # Initialize architecture
    nn_arch = architecture_dict[architecture](channels=3)
    # Define input shape
    input_shape = nn_arch.input[:-1]

    # Initialize model
    model = Neural_Network(config["nclasses"], channels=3, architecture=nn_arch,
                           workers=config["workers"], multiprocessing=False,
                           batch_queue_size=config["batch_queue_size"],
                           activation_output=activation_output,
                           loss="categorical_crossentropy",
                       