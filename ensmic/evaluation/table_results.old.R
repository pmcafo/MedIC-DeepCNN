# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("dplyr")

# Configurations
path_results <- "/home/mudomini/projects/ensmic/results/"
datasets <- c("chmnist", "covid", "isic", "drd")
phases <- c("baseline", "augmenting", "stacking", "bagging")

path_eval <- file.path(path_results, "evaluation")
dir.create(path_eval, showWarnings=FALSE)

# Gather results for baseline, augmenting & stacing
for (i in seq(1,3)){
  dt <- data.table()
  
  for (j in seq(1,4)){
    path_dir <-