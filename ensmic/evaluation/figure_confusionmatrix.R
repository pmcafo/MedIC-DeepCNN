# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("grid")
library("gridExtra")
library("stringr")

# Configurations
path_results <- "/home/mudomini/projects/ensmic/results/"
datasets <- c("chmnist", "covid", "isic", "drd")
phases <- c("baseline", "augmenting", "stacking", "bagging")

path_eval <- file.path(path_results, "evaluation")
dir.create(path_eval, showWarnings=FALSE)

# Load data
dt <- fread(file.path(path_results, "eval_tmp", "confusion_matrix.csv"))


# Preprocess
dt$dataset <- paste("Dataset:", dt$dataset)
dt <- transform(dt, dataset=factor(dataset, levels=c("Dataset: CHMNIST","Dataset: COVID",
                                