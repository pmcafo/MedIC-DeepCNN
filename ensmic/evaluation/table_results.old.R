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
    path_dir <- file.path(path_results, paste("phase_", phases[i], ".", datasets[j], sep = "", collapse = NULL))
    path_file <- file.path(path_dir, "evaluation", "results.test.averaged.csv")
    dt_tmp <- fread(path_file)
    dt_tmp[, dataset:=datasets[j]]
    if("ensembler" %in% colnames(dt_tmp))
    {
      setnames(dt_tmp, "ensembler", "model")
    }
    if("architecture" %in% colnames(dt_tmp))
    {
      setnames(dt_tmp, "architecture", "model")
    }
    dt_tmp <- dt_tmp[metric %in% c("Accuracy", "F1", "Sensitivity", "Speci