args <- commandArgs(trailingOnly = TRUE)

normalize_rows <- function(mat) {
  sums <- rowSums(mat)
  for (i in seq_len(nrow(mat))) {
    if (sums[i] > 0) {
      mat[i, ] <- mat[i, ] / sums[i]
    }
  }
  mat
}

stationary_distribution <- function(P) {
  eigen_result <- eigen(t(P))
  idx <- which.min(abs(eigen_result$values - 1))
  vec <- Re(eigen_result$vectors[, idx])
  vec / sum(vec)
}

hitting_times <- function(P, target) {
  n <- nrow(P)
  idx <- setdiff(seq_len(n), target)
  Q <- P[idx, idx, drop = FALSE]
  h <- solve(diag(length(idx)) - Q, rep(1, length(idx)))
  out <- rep(0, n)
  out[idx] <- h
  out
}

viterbi_decode <- function(obs, transition, emission, initial = NULL) {
  n_states <- nrow(transition)
  if (is.null(initial)) {
    initial <- rep(1 / n_states, n_states)
  }
  log_delta <- log(initial + 1e-12) + log(emission[, obs[1] + 1] + 1e-12)
  psi <- matrix(0, nrow = length(obs), ncol = n_states)
  for (t in 2:length(obs)) {
    scores <- outer(log_delta, rep(1, n_states)) + log(transition + 1e-12)
    psi[t, ] <- apply(scores, 2, which.max)
    log_delta <- apply(scores, 2, max) + log(emission[, obs[t] + 1] + 1e-12)
  }
  states <- integer(length(obs))
  states[length(obs)] <- which.max(log_delta)
  for (t in (length(obs) - 1):1) {
    states[t] <- psi[t + 1, states[t + 1]]
  }
  states - 1
}

baum_welch_step <- function(obs, transition, emission, initial = NULL) {
  n_states <- nrow(transition)
  n_obs <- ncol(emission)
  if (is.null(initial)) {
    initial <- rep(1 / n_states, n_states)
  }
  Tn <- length(obs)
  alpha <- matrix(0, nrow = Tn, ncol = n_states)
  beta <- matrix(0, nrow = Tn, ncol = n_states)
  scale <- rep(0, Tn)

  alpha[1, ] <- initial * emission[, obs[1] + 1]
  scale[1] <- sum(alpha[1, ]) + 1e-12
  alpha[1, ] <- alpha[1, ] / scale[1]
  for (t in 2:Tn) {
    alpha[t, ] <- (alpha[t - 1, ] %*% transition) * emission[, obs[t] + 1]
    scale[t] <- sum(alpha[t, ]) + 1e-12
    alpha[t, ] <- alpha[t, ] / scale[t]
  }

  beta[Tn, ] <- 1
  for (t in (Tn - 1):1) {
    beta[t, ] <- transition %*% (emission[, obs[t + 1] + 1] * beta[t + 1, ])
    beta[t, ] <- beta[t, ] / (sum(beta[t, ]) + 1e-12)
  }

  gamma <- alpha * beta
  gamma <- gamma / rowSums(gamma)

  xi_sum <- matrix(0, nrow = n_states, ncol = n_states)
  for (t in 1:(Tn - 1)) {
    xi <- transition * outer(alpha[t, ], emission[, obs[t + 1] + 1] * beta[t + 1, ])
    xi <- xi / (sum(xi) + 1e-12)
    xi_sum <- xi_sum + xi
  }

  new_transition <- normalize_rows(xi_sum)
  new_emission <- matrix(0, nrow = n_states, ncol = n_obs)
  for (k in 0:(n_obs - 1)) {
    mask <- which(obs == k)
    if (length(mask) > 0) {
      new_emission[, k + 1] <- colSums(gamma[mask, , drop = FALSE])
    }
  }
  new_emission <- normalize_rows(new_emission)
  list(transition = new_transition, emission = new_emission, gamma = gamma)
}

run_demo <- function() {
  print("HELIOS R Markov analysis ready")
}

if (length(args) < 1) {
  run_demo()
} else {
  output_path <- args[1]
  states <- c(0, 1, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2)
  n_states <- max(states) + 1
  counts <- matrix(0, nrow = n_states, ncol = n_states)
  for (i in 1:(length(states) - 1)) {
    counts[states[i] + 1, states[i + 1] + 1] <- counts[states[i] + 1, states[i + 1] + 1] + 1
  }
  P <- normalize_rows(counts)
  stat <- stationary_distribution(P)
  hits <- hitting_times(P, 3)
  absorb <- matrix(c(0.5, 0.5, 0, 0.2, 0.5, 0.3, 0, 0, 1), nrow = 3, byrow = TRUE)
  absorb_hits <- hitting_times(absorb, 3)

  obs <- c(0, 0, 1, 1, 1, 0)
  transition <- matrix(c(0.85, 0.15, 0.2, 0.8), nrow = 2, byrow = TRUE)
  emission <- matrix(c(0.9, 0.1, 0.2, 0.8), nrow = 2, byrow = TRUE)
  decoded <- viterbi_decode(obs, transition, emission)
  bw <- baum_welch_step(obs, transition, emission)
  decoded_short <- decoded[seq_len(min(length(decoded), n_states))]
  padded_decoded <- c(decoded_short, rep(NA, max(n_states - length(decoded_short), 0)))

  out <- data.frame(
    state = 0:(n_states - 1),
    stationary_probability = stat,
    hitting_time_to_state_2 = hits,
    absorbing_hitting_time = absorb_hits,
    viterbi_state = padded_decoded[1:n_states],
    baum_welch_t00 = bw$transition[1, 1],
    baum_welch_t01 = bw$transition[1, 2]
  )
  write.csv(out, output_path, row.names = FALSE)
}
