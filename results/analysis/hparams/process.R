data <- read.csv("/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/rawdata.csv", header = T)
processed <- aggregate(.~+id2, data, mean)