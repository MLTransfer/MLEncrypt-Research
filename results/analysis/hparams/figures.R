data <-
  read.csv(
    "https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/hparams/rawdata.csv",
    header = T
  )
data$update_rule = gsub("anti_hebbian", "0", data$update_rule)
data$update_rule = gsub("hebbian", "-1", data$update_rule)
data$update_rule = gsub("random_walk", "1", data$update_rule)
data$attack = gsub("geometric", "1", data$attack)
data$attack = gsub("none", "-1", data$attack)
library(plotly)
size = list(titlefont = list(size = 32))
pcp <- data %>%
  plot_ly(width = 1000, height = 600) %>%
  layout(title = "Parallel Coordinates Plot of Hyperparameter Data",
         scene = list(xaxis = size,
                      yaxis = size)) %>%
  add_trace(
    type = 'parcoords',
    line = list(
      color = ~ time_taken,
      colorscale = 'Viridis',
      showscale = TRUE,
      cmin = ~ min(time_taken),
      cmax = ~ max(time_taken)
    ),
    dimensions = list(
      list(
        range = c(0, 24),
        visible = TRUE,
        label = 'K',
        values = ~ K
      ),
      list(
        range = c(0, 24),
        visible = TRUE,
        label = 'N',
        values = ~ N
      ),
      list(
        range = c(0, 24),
        visible = TRUE,
        label = 'L',
        values = ~ L
      ),
      list(
        tickvals = c(-1, 0, 1),
        ticktext = c('Hebbian', 'anti-Hebbian', 'random walk'),
        label = 'Update Rule',
        values = ~ update_rule
      ),
      list(
        tickvals = c(-1, 1),
        ticktext = c('none', 'geometric'),
        label = 'Attack',
        values = ~ attack
      ),
      list(
        range = c( ~ min(time_taken),  ~ max(time_taken)),
        label = 'Training Time (s)',
        values = ~ time_taken
      ),
      list(
        range = c( ~ min(eve_score),  ~ max(eve_score)),
        label = 'Eve\'s score (%)',
        values = ~ eve_score
      )
    )
  )
pcp  # parallel coordinates plot


data$KN = data$K * data$N
axis_x <- seq(min(data$KN), max(data$KN))
axis_y <- seq(min(data$L), max(data$L))
lm_t <- lm(time_taken ~ KN + L, data = data)
lm_t_surface <- expand.grid(KN = axis_x,
                            L = axis_y,
                            KEEP.OUT.ATTRS = F)
lm_t_surface$time_taken <- predict.lm(lm_t, newdata = lm_t_surface)
lm_t_surface <-
  acast(lm_t_surface, L ~ KN, value.var = "time_taken")
s_t <-
  plot_ly(
    data,
    x = ~ KN,
    y = ~ L,
    z = ~ time_taken,
    type = 'scatter3d',
    mode = 'lines+markers+text'
  ) %>%
  layout(scene = list(
    xaxis = list(title = 'KN'),
    yaxis = list(title = 'L'),
    zaxis = list(title = 'Training Time (s)')
  ))
s_t <- add_trace(
  p = s_t,
  z = lm_t_surface,
  x = axis_x,
  y = axis_y,
  type = "surface"
)
s_t  # scatterplot of time vs KN + L
lm_e <- lm(eve_score ~ KN + L, data = data)
lm_e_surface <- expand.grid(KN = axis_x,
                            L = axis_y,
                            KEEP.OUT.ATTRS = F)
lm_e_surface$eve_score <- predict.lm(lm_e, newdata = lm_e_surface)
lm_e_surface <-
  acast(lm_e_surface, L ~ KN, value.var = "eve_score")
s_e <-
  plot_ly(
    data,
    x = ~ KN,
    y = ~ L,
    z = ~ eve_score,
    type = 'scatter3d',
    mode = 'lines+markers+text'
  ) %>%
  layout(scene = list(
    xaxis = list(title = 'KN'),
    yaxis = list(title = 'L'),
    zaxis = list(title = 'Eve\'s Score (%)')
  ))
s_e <- add_trace(
  p = s_e,
  z = lm_e_surface,
  x = axis_x,
  y = axis_y,
  type = "surface"
)
s_e
