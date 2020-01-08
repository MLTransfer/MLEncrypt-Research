# https://drsimonj.svbtle.com/pretty-scatter-plots-with-ggplot2

data <-
  read.csv(file = "https://raw.githubusercontent.com/MLTransfer/MLEncrypt-Research/master/results/analysis/training-time/training_time.csv", header = T)

data$pc <- predict(prcomp( ~ steps + training_time, data))[, 1]

# Add density for each point
data$density <-
  fields::interp.surface(MASS::kde2d(data$steps, data$training_time), data[, c("steps", "training_time")])

library(ggplot2)
s_tt <-
  ggplot(data, aes(
    steps,
    training_time,
    color = pc,
    alpha = 1 / density
  )) +
  geom_point(size = 3) + theme_minimal() +
  scale_color_gradient(low = "#32aeff", high = "#f2aeff") +
  scale_alpha(range = c(.25, .6)) +
  xlab("Number of steps") +
  ylab("Training Time (s)") +
  scale_alpha_continuous(name = "Bivariate Density") +
  scale_color_continuous(name = "PCA")
s_tt  # scatterplot of training time vs. number of training steps