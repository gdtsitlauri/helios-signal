args <- commandArgs(trailingOnly = TRUE)

fit_garch11 <- function(residuals) {
  residuals <- as.numeric(residuals)
  objective <- function(par) {
    omega <- abs(par[1]) + 1e-6
    alpha <- 1 / (1 + exp(-par[2]))
    beta <- 1 / (1 + exp(-par[3]))
    if (alpha + beta >= 0.999) {
      return(1e9)
    }
    h <- rep(var(residuals) + 1e-6, length(residuals))
    for (i in 2:length(residuals)) {
      h[i] <- omega + alpha * residuals[i - 1]^2 + beta * h[i - 1]
    }
    sum(log(h) + residuals^2 / h)
  }
  opt <- optim(c(0.1, 0, 0), objective)
  c(
    omega = abs(opt$par[1]) + 1e-6,
    alpha = 1 / (1 + exp(-opt$par[2])),
    beta = 1 / (1 + exp(-opt$par[3]))
  )
}

granger_score <- function(x, y, lag = 1) {
  x <- as.numeric(x)
  y <- as.numeric(y)
  if (length(x) <= lag + 2 || length(y) <= lag + 2) {
    return(0)
  }
  target <- y[(lag + 1):length(y)]
  y_lag <- y[1:(length(y) - lag)]
  x_lag <- x[1:(length(x) - lag)]
  base_fit <- lm(target ~ y_lag)
  full_fit <- lm(target ~ y_lag + x_lag)
  rss_base <- mean(residuals(base_fit)^2)
  rss_full <- mean(residuals(full_fit)^2)
  max((rss_base - rss_full) / (rss_base + 1e-12), 0)
}

engle_granger_score <- function(x, y) {
  fit <- lm(y ~ x)
  res <- residuals(fit)
  dres <- diff(res)
  res_lag <- res[-length(res)]
  adf_fit <- lm(dres ~ res_lag)
  coef(adf_fit)[["res_lag"]]
}

run_demo <- function() {
  demo_series <- cos(seq(0, 2 * pi, length.out = 64))
  fit <- arima(demo_series, order = c(1, 0, 0))
  print("HELIOS R ARIMA bridge ready")
  print(fit$coef)
}

if (length(args) < 3) {
  run_demo()
} else {
  input_path <- args[1]
  output_path <- args[2]
  horizon <- as.integer(args[3])

  frame <- read.csv(input_path)
  values <- as.numeric(frame$signal)
  aux <- if ("aux" %in% names(frame)) as.numeric(frame$aux) else c(values[-1], values[length(values)])

  arima_fit <- arima(values, order = c(1, 0, 0))
  arima_forecast <- predict(arima_fit, n.ahead = horizon)$pred

  seasonal_fit <- tryCatch(
    arima(ts(values, frequency = 12), order = c(1, 0, 0), seasonal = list(order = c(1, 0, 0), period = 12)),
    error = function(e) NULL
  )
  seasonal_forecast <- if (!is.null(seasonal_fit)) {
    as.numeric(predict(seasonal_fit, n.ahead = horizon)$pred)
  } else {
    rep(NA_real_, horizon)
  }

  residuals_arima <- residuals(arima_fit)
  garch_params <- fit_garch11(residuals_arima)
  spectrum <- spec.pgram(values, plot = FALSE)
  top_idx <- which.max(spectrum$spec)
  granger <- granger_score(aux, values, lag = 1)
  coint <- engle_granger_score(aux, values)

  out <- data.frame(
    horizon = seq_len(horizon),
    forecast_arima = as.numeric(arima_forecast),
    forecast_sarima = seasonal_forecast,
    garch_omega = garch_params[["omega"]],
    garch_alpha = garch_params[["alpha"]],
    garch_beta = garch_params[["beta"]],
    granger_score = granger,
    cointegration_score = coint,
    peak_frequency = spectrum$freq[top_idx],
    peak_power = spectrum$spec[top_idx]
  )
  write.csv(out, output_path, row.names = FALSE)
}
