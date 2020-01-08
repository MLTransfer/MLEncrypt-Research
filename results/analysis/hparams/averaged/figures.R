data <-
  read.csv(
    # "/Users/suman/quantum/mltransfer/mlencrypt-research/results/analysis/hparams/averaged/averaged.csv",
    "https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/hparams/averaged/averaged.csv",
    header = T
  )
data$update_rule = gsub("anti_hebbian", 0, data$update_rule)
data$update_rule = gsub("hebbian",-1, data$update_rule)
data$update_rule = gsub("random_walk", 1, data$update_rule)
library(plotly)
size = list(size = 32)
pcp <- data %>%
  plot_ly(width = 1000, height = 600) %>%
  layout(title = "Parallel Coordinates Plot of Averaged Hyperparameter Data",
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
        range = c( ~ min(time_taken),  ~ max(time_taken)),
        label = 'Training Time (s)',
        values = ~ time_taken
      ),
      list(
        range = c( ~ min(eve_score_none),  ~ max(eve_score_none)),
        label = 'Eve\'s score (%), no attack',
        values = ~ eve_score_none
      ),
      list(
        range = c( ~ min(eve_score_geometric),  ~ max(eve_score_geometric)),
        label = 'Eve\'s score (%), geometric',
        values = ~ eve_score_geometric
      ),
      list(
        range = c( ~ min(eve_score_average),  ~ max(eve_score_average)),
        label = 'Eve\'s score (%), average',
        values = ~ eve_score_average
      )
    )
  )
pcp  # parallel coordinates plot

library(reshape2)
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

lm_e <- lm(eve_score_average ~ KN + L, data = data)
lm_e_surface <- expand.grid(KN = axis_x,
                            L = axis_y,
                            KEEP.OUT.ATTRS = F)
lm_e_surface$eve_score_average <-
  predict.lm(lm_e, newdata = lm_e_surface)
lm_e_surface <-
  acast(lm_e_surface, L ~ KN, value.var = "eve_score_average")
s_e <-
  plot_ly(
    data,
    x = ~ KN,
    y = ~ L,
    z = ~ eve_score_average,
    type = 'scatter3d',
    mode = 'lines+markers+text'
  ) %>%
  layout(scene = list(
    xaxis = list(title = 'KN'),
    yaxis = list(title = 'L'),
    zaxis = list(title = 'Eve\'s Score (%), average')
  ))
s_e <- add_trace(
  p = s_e,
  z = lm_e_surface,
  x = axis_x,
  y = axis_y,
  type = "surface"
)
s_e

ystats = boxplot.stats(data$time_taken)$stats
acceptable = 1.5 * (ystats[4] - ystats[2])
ymax = ystats[4] + acceptable
ymin = ystats[2] - acceptable
nooutliers = subset(data, time_taken < ymax)  # remove outliers

# https://drsimonj.svbtle.com/pretty-scatter-plots-with-ggplot2
nooutliers$pc <- predict(prcomp( ~ L + time_taken, nooutliers))[, 1]

# Add density for each point
nooutliers$density <-
  fields::interp.surface(MASS::kde2d(nooutliers$L, nooutliers$time_taken),
                         nooutliers[, c("L", "time_taken")])

s_st <-
  ggplot(nooutliers, aes(L, time_taken, color = pc, alpha = 1 / density)) +
  geom_point(size = 3) + theme_minimal() +
  scale_color_gradient(low = "#32aeff", high = "#f2aeff") +
  scale_alpha(range = c(.25, .6)) +
  xlab("L") +
  ylab("Training Time (s)") +
  scale_alpha_continuous(name = "Bivariate Density") +
  scale_color_continuous(name = "PCA") +
  stat_smooth(
    method = "lm",
    formula = y ~ poly(x, 2),
    size = 1,
    show.legend = FALSE
  )

s_st  # scatterplot of training time vs L
