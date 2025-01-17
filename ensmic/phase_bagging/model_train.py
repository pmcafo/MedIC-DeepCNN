
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
import numpy as np
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
# AUCMEDI libraries
from aucmedi import DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode, \
                                                 architecture_dict
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.utils.class_weights import compute_class_weights
# ENSMIC libraries
from ensmic.data_loading import load_sampling

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="ENSMIC: Phase I - Training")
parser.add_argument("-m", "--modularity", help="Data modularity selection: ['covid', 'isic', 'chmnist', 'drd']",
                    required=True, type=str, dest="seed")
parser.add_argument("-a", "--architecture", help="Architecture on which CV/bagging is performed",
                    required=True, type=str, dest="architecture")
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
config["batch_size"] = 28
config["batch_queue_size"] = 16
# Neural Network Configurations
config["epochs"] = 1000
config["iterations"] = None
config["workers"] = 32
config["architecture"] = args.architecture

# Cross-Validation Configurations
config["k_fold"] = 5

# Adjust GPU configuration
config["gpu_id"] = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])

#-----------------------------------------------------#
#                   AUCMEDI Pipeline                  #
#-----------------------------------------------------#
def run_aucmedi(fold, architecture, config):
    # Create result directory for architecture & cv fold
    path_arch = os.path.join(config["path_phase"], architecture)
    if not os.path.exists(path_arch) : os.mkdir(path_arch)
    path_cv = os.path.join(path_arch, "cv_" + str(fold))
    if not os.path.exists(path_cv) : os.mkdir(path_cv)

    # If latest model for current fold already exist -> skip
    if os.path.exists(os.path.join(path_cv, "model.last.hdf5")):
        return

    # Load sampling from disk
    sampling_train = load_sampling(path_input=config["path_data"],
                                   subset="cv" + str(fold) + "_train",
                                   seed=config["seed"])
    (x_train, y_train, nclasses, _, image_format) = sampling_train
    sampling_val = load_sampling(path_input=config["path_data"],
                                 subset="cv" + str(fold) + "_val",
                                 seed=config["seed"])
    (x_val, y_val, _, _, _) = sampling_val

    # Compute classweights
    _, class_weights_dict = compute_class_weights(y_train)

    # Initialize Image Augmentation
    aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                             saturation=True, hue=True, scale=False, crop=False,
                             grid_distortion=False, compression=False, gamma=False,
                             gaussian_noise=False, gaussian_blur=False,
                             downscaling=False, elastic_transform=False)
    # Define Subfunctions
    sf_list = [Padding(mode="square")]
    # Set activation output to softmax for multi-class classification
    activation_output = "softmax"

    # Initialize architecture
    nn_arch = architecture_dict[architecture](channels=3)
    # Define input shape
    input_shape = nn_arch.input[:-1]

    # Initialize model
    model = Neural_Network(nclasses, channels=3, architecture=nn_arch,
                           workers=config["workers"], multiprocessing=False,
                           batch_queue_size=config["batch_queue_size"],
                           activation_output=activation_output,
                           loss="categorical_crossentropy",
                           metrics=[CategoricalAccuracy(), AUC(100)],
                           pretrained_weights=True)
    # Modify number of transfer learning epochs with frozen model layers
    model.tf_epochs = 10

    # Obtain standardization mode for current architecture
    sf_standardize = supported_standardize_mode[architecture]

    # Initialize training and validation Data Generators
    train_gen = DataGenerator(x_train, config["path_images"], labels=y_train,
                              batch_size=config["batch_size"], img_aug=aug,
                              shuffle=True, subfunctions=sf_list,
                              resize=input_shape, standardize_mode=sf_standardize,
                              grayscale=False, prepare_images=False,
                              sample_weights=None, seed=None,
                              image_format=image_format,
                              workers=config["threads"])
    val_gen = DataGenerator(x_val, config["path_images"], labels=y_val,
                            batch_size=config["batch_size"], img_aug=None,
                            subfunctions=sf_list, shuffle=False,
                            standardize_mode=sf_standardize, resize=input_shape,
                            grayscale=False, prepare_images=False, seed=None,
                            sample_weights=None, workers=config["threads"],
                            image_format=image_format)

    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join(path_cv, "model.best.hdf5"),
                            monitor="val_loss", verbose=1,
                            save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join(path_cv, "logs.csv"), separator=',', append=True)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='min', min_lr=1e-7)
    cb_es = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

    # Train model
    model.train(train_gen, val_gen, class_weights=class_weights_dict,
                epochs=config["epochs"], iterations=config["iterations"],
                callbacks=callbacks, transfer_learning=True)

    # Dump latest model
    model.dump(os.path.join(path_cv, "model.last.hdf5"))

    # Garbage collection
    del train_gen
    del val_gen
    del model

#-----------------------------------------------------#
#                     Main Runner                     #
#-----------------------------------------------------#
# Create results directory
if not os.path.exists(config["path_results"]) : os.mkdir(config["path_results"])
# Create subdirectories for phase & seed
config["path_phase"] = os.path.join(config["path_results"], "phase_bagging" + "." + \
                                    str(config["seed"]))
if not os.path.exists(config["path_phase"]) : os.mkdir(config["path_phase"])

# Define imaging directory
config["path_images"] = os.path.join(config["path_data"],
                                     config["seed"] + ".images")

# Initialize cache memory to store meta information
timer_cache = {}

# Access architecture
architecture = config["architecture"]

# Run Fitting Pipeline for each fold in the CV
for fold in range(0, config["k_fold"]):
    print("Run Training for Architecture:", architecture, " - Fold " + str(fold))
    timer_start = time.time()
    run_aucmedi(fold, architecture, config)
    timer_end = time.time()
    # Store execution time in cache
    timer_time = timer_end - timer_start
    timer_cache[architecture + ".cv_" + str(fold)] = timer_time
    print("Finished Training for Architecture:", architecture, " - Fold " + str(fold))

# Store time measurements as JSON to disk
path_time = os.path.join(config["path_phase"], "time_measurements." + "architecture_" + str(architecture) + ".json")
with open(path_time, "w") as file:
    json.dump(timer_cache, file, indent=2)