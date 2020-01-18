data <-
  read.csv(
    # "/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/averaged/averaged.csv",
    "https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/hparams/averaged/averaged.csv",
    header = T
  )

library(dplyr)
library(rPref)
p <-
  low(adversary_score_none) *
  low(adversary_score_geometric) *
  low(training_time)
res <- psel(data, p)
assoc.df(p) <- data
res <- peval(p)
knitr::kable(select(
  res,
  adversary_score_none,
  adversary_score_geometric,
  training_time
))

library(plotly)
skyline <-
  plot_ly(type = "scatter3d", mode = "markers") %>%
  layout(scene = list(
    xaxis = list(title = "Adversary Score (%), no attack"),
    yaxis = list(title = "Adversary Score (%), geometric"),
    zaxis = list(title = "Training Time (s)")
  )) %>%
  add_trace(
    data = data,
    x = ~ adversary_score_none,
    y = ~ adversary_score_geometric,
    z = ~ training_time,
    name = "All data"
  ) %>%
  add_trace(
    data = res,
    x = ~ adversary_score_none,
    y = ~ adversary_score_geometric,
    z = ~ training_time,
    name = "Pareto frontier"
  ) %>%
  add_trace(
    data = res[order(res$adversary_score_none,
                     res$adversary_score_geometric,
                     res$training_time), ],
    x = ~ adversary_score_none,
    y = ~ adversary_score_geometric,
    z = ~ training_time,
    type = "mesh3d"
  )
skyline

# plot_btg(data, p)
