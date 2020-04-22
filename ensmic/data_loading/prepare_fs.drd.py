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
import pandas as pd
import os
import json
from shutil import copyfile
from tqdm import tqdm
# AUCMEDI libraries
from aucmedi.sampling import sampling_split
from aucmedi import input_interface
# Internal libraries/scripts
from ensmic.data_loading import sampling_to_disk

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# File structure
path_input = "data.drd"
path_target = "data"

# Sampling strategy (in percentage)
sampling_splits = [0.65, 0.10, 0.10, 0.15]
sampling_names = ["train-model", "val-model", "val-ensemble", "test"]
# P