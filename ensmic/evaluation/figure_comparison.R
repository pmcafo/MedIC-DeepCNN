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
dt_proc$phase <- factor(dt_proc$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt$phase <- factor(dt$phase, levels=c("Baseline", "Augmenting", "Bagging", "Stacking"))
dt_proc$dataset <- factor(dt_proc$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))
dt$dataset <- factor(dt$dataset, levels=c("CHMNIST", "COVID", "ISIC", "DRD"))

# Plot comparison
dodge <- position_dodge(width=0.8)
plot_comparison <- ggplot(dt_proc, aes(x=dataset, y=max, fill=phase)) +
  geom_bar(stat="identity", position=dodge, color="black", width=0.4, alpha=0.4) +
  stat_boxplot(data=dt, aes(x=dataset, y=F1), position=dodge, geom="errorbar") +
  geom_boxplot(data=dt, aes(x=dataset, y=F1), position=dodge, outlier.shape=NA) +
  geom_text(aes(y=0, label=phase), position=dodge, hjust=-0.2, vjust=0.4, angle=90) +
  scale_fill_brewer(palette="Set1") +
  scale_y_continuous(breaks=seq(0, 1, 0.05), limits=c(0, 1)) +
  theme_bw() +
  theme(legend.position = "none") +
  xlab("") +
  ylab("Score (Accuracy / F1)") +
  ggtitle("Comparison of Ensemble Learning Performance Influence on multiple datasets")
png(file.path(path_eval, "figure.comparison.png"), 