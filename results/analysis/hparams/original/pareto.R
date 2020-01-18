data <-
  read.csv(
    # "/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/original/rawdata.csv",
    "https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/hparams/original/rawdata.csv",
    header = T
  )

library(dplyr)
library(rPref)
p <-  low(training_time) * low(adversary_score)
res <- psel(data, p)
assoc.df(p) <- data
res <- peval(p)
knitr::kable(select(res, training_time, adversary_score))

library(plotly)
skyline <-
  plot_ly(type = "scatter", mode = "markers") %>%
  layout(
    xaxis = list(title = "Training Time (s)"),
    yaxis = list(title = "Adversary Score (%)")
  ) %>%
  add_trace(
    data = data,
    x = ~ training_time,
    y = ~ adversary_score,
    name = "All data"
  ) %>%
  add_trace(
    data = res,
    x = ~ training_time,
    y = ~ adversary_score,
    name = "Pareto frontier"
  )
skyline

# plot_btg(data, p)
