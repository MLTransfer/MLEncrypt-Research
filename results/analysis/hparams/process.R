data <- read.csv("https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/hparams/rawdata.csv", header = T)
processed <- aggregate(.~+id2, data, mean)