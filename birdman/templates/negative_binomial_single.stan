data {
  int<lower=0> N;       // number of samples
  int<lower=0> p;       // number of covariates
  real depth[N];        // sequencing depths of microbe
  matrix[N, p] x;       // covariate matrix
  int y[N];             // observed microbe abundances
  real<lower=0> B_p;    // stdev for Beta Normal prior
  real<lower=0> phi_s;  // scale for dispersion Cauchy prior
}

parameters {
  // parameters required for linear regression on the species means
  vector[p] beta;
  real reciprocal_phi;
}

transformed parameters {
  vector[N] lam;
  real phi;

  phi = 1. / reciprocal_phi;
  lam = x * beta;
}

model {
  // setting priors ...
  reciprocal_phi ~ cauchy(0., phi_s);
  for (j in 1:p){
    beta[j] ~ normal(0., B_p); // uninformed prior
  }
  // generating counts
  for (n in 1:N){
    target += neg_binomial_2_log_lpmf(y[n] | depth[n] + lam[n], phi);
  }
}
