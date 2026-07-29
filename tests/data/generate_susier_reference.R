# Regenerate susier_v0.11.92_reference.json with the exact pinned R package.
# Run with an R library containing susieR 0.11.92 and jsonlite, then redirect
# stdout to the JSON file.

library(jsonlite)
library(susieR)

stopifnot(as.character(packageVersion("susieR")) == "0.11.92")

p <- 8L
rho <- 0.6
R <- outer(seq_len(p), seq_len(p), function(i, j) rho^abs(i - j))
diag(R) <- 1
z <- c(0.2, -0.4, 5.5, 0.1, -0.2, -4.2, 0.3, 0.0)

fit <- susie_suff_stat(
  bhat = z,
  shat = rep(1, p),
  R = R,
  n = 5000L,
  L = 3L,
  scaled_prior_variance = 1e-4,
  residual_variance = NULL,
  estimate_prior_variance = TRUE,
  estimate_residual_variance = TRUE,
  max_iter = 100L,
  standardize = FALSE,
  prior_weights = NULL,
  coverage = 0.95,
  min_abs_corr = 0.5,
  tol = 1e-3,
  prior_tol = 1e-9,
  n_purity = 100L
)

fixture <- list(
  provenance = list(
    susieR_version = as.character(packageVersion("susieR")),
    susieR_tag = "v0.11.92",
    susieR_commit = "23606070c3025584a5e5cbd0cb6c7abb2fd4c4d4",
    R_version = R.version.string,
    generated = format(Sys.Date(), "%Y-%m-%d")
  ),
  input = list(z = z, R = R, n = 5000L, L = 3L),
  parameters = list(
    scaled_prior_variance = 1e-4,
    estimate_prior_variance = TRUE,
    estimate_residual_variance = TRUE,
    max_iter = 100L,
    standardize = FALSE,
    coverage = 0.95,
    min_abs_corr = 0.5,
    tol = 1e-3,
    prior_tol = 1e-9,
    n_purity = 100L
  ),
  expected = list(
    alpha = unclass(fit$alpha),
    mu = unclass(fit$mu),
    mu2 = unclass(fit$mu2),
    V = unclass(fit$V),
    sigma2 = unclass(fit$sigma2),
    pip = unclass(susie_get_pip(fit, prune_by_cs = FALSE, prior_tol = 1e-9)),
    elbo = unclass(fit$elbo),
    niter = unclass(fit$niter),
    converged = unclass(fit$converged),
    sets = unname(lapply(fit$sets$cs, function(x) I(unname(x - 1L)))),
    cs_index = unname(fit$sets$cs_index - 1L)
  )
)

cat(
  toJSON(
    fixture,
    auto_unbox = TRUE,
    digits = 17,
    pretty = TRUE,
    matrix = "rowmajor"
  )
)
cat("\n")
