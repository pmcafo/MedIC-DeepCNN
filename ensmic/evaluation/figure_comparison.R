# Import libraries
library("data.table")
library("ggplot2")
library("tidyr")
library("stringr")

# Configurations
path_eval <- "/home/mudomini/projects/ensmic/results/evaluation/"
datasets <- c("baseline", "augmenting", "bagging", "stacking")

# Gather data
dt <- data.table()
for (ds in datasets){
  path_file <- file.path(path_eval, paste("table", "results", ds, "csv", sep="."))
  dt_tmp <- fread(path_file)
  dt_tmp[, phase:=ds]
  dt <- rbind(dt, dt_tmp)
}

# Identify best methods & merge
dt$dataset <- sapply(dt$dataset, toupper)
dt$phase <- sapply(dt$phase, str_to_title)
dt_proc <- dt[, .(min=min(Accuracy), max=max(Accuracy)), by=list(phase, dataset)]

# Order labels
dt_proc$phase <- factor(dt_proc$phase, levels=c("Baseline", "Augmenting", "Bagging", 