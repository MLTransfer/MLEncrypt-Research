# https://drsimonj.svbtle.com/pretty-scatter-plots-with-ggplot2

data <-
  read.csv(file = "/Users/suman/quantum/mltransfer/mlencrypt-research/results/time_taken.csv", header = T)

data$pc <- predict(prcomp( ~ steps + time.taken, data))[, 1]

# Add density for each point
data$density <-
  fields::interp.surface(MASS::kde2d(data$steps, data$time.taken), data[, c("steps", "time.taken")])

# Plot
ggplot(data, aes(steps, time.taken, color = pc, alpha = 1 / density)) +
  geom_point(size = 3) + theme_minimal() +
  scale_color_gradient(low = "#32aeff", high = "#f2aeff") +
  scale_alpha(range = c(.25, .6)) +
  xlab("Number of stepss") +
  ylab("Training Time") +
  scale_alpha_continuous(name = "Bivariate Density") +
  scale_color_continuous(name = "PCA")
