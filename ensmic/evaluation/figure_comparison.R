# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("stringr")

# Configurations
path_eval <- "/home/mudomini/projects/ensmic/results/evaluation/"
datasets <- c("baseline", "augmenting", "bagging", "stacking")

# Gat