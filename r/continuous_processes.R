args <- commandArgs(trailingOnly = TRUE)

simulate_poisson <- function(rate, horizon) {
  t <- 0
  out <- c()
  while (t < horizon) {
    t <- t + rexp(1, rate = rate)
    if (t <= horizon) {
      out <- c(out, t)
    }
  }
  out
}

simulate_brownian <- function(steps, dt) {
  c(0, cumsum(rnorm(steps, mean = 0, sd = sqrt(dt))))
}

simulate_gbm <- function(s0, mu, sigma, steps, dt) {
  w <- simulate_brownian(steps, dt)
  t <- seq(0, steps) * dt
  s0 * exp((mu - 0.5 * sigma^2) * t + sigma * w)
}

simulate_ou <- function(theta, mu, sigma, steps, dt) {
  x <- rep(0, steps + 1)
  for (i in 1:steps) {
    x[i + 1] <- x[i] + theta * (mu - x[i]) * dt + sigma * sqrt(dt) * rnorm(1)
  }
  x
}

euler_maruyama <- function(x0, drift, diffusion, steps, dt) {
  x <- rep(0, steps + 1)
  x[1] <- x0
  for (i in 1:steps) {
    x[i + 1] <- x[i] + drift(x[i]) * dt + diffusion(x[i]) * sqrt(dt) * rnorm(1)
  }
  x
}

run_demo <- function() {
  print("HELIOS R continuous process simulations ready")
}

if (length(args) < 1) {
  run_demo()
} else {
  output_path <- args[1]
  set.seed(0)
  poisson <- simulate_poisson(2.0, 5.0)
  brownian <- simulate_brownian(128, 0.05)
  gbm <- simulate_gbm(1.0, 0.05, 0.2, 128, 1 / 252)
  ou <- simulate_ou(0.7, 0.0, 0.2, 128, 0.1)
  sde <- euler_maruyama(0.0, function(x) -0.3 * x, function(x) 0.15, 128, 0.1)

  out <- data.frame(
    process = c("poisson_count", "brownian_last", "gbm_last", "ou_last", "euler_maruyama_last"),
    value = c(length(poisson), tail(brownian, 1), tail(gbm, 1), tail(ou, 1), tail(sde, 1))
  )
  write.csv(out, output_path, row.names = FALSE)
}
