data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  int<lower=0> p;       // number of covariates
  real depth[N];        // sequencing depths of microbes
  matrix[N, p] x;       // covariate matrix
  int y[N, D];          // observed microbe abundances
  real<lower=0> B_p;    // stdev for Beta Normal prior
  real<lower=0> phi_s;  // scale for dispersion Cauchy prior
}

parameters {
  // parameters required for linear regression on the species means
  matrix[p, D-1] beta;
  real reciprocal_phi;
}

transformed parameters {
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;
  matrix[N, D] prob;
  vector[N] z;
  real phi;

  phi = 1. / reciprocal_phi;

  z = to_vector(rep_array(0, N));
  lam = x * beta;
  lam_clr = append_col(z, lam);
}

model {
  // setting priors ...
  reciprocal_phi ~ cauchy(0., phi_s);
  for (i in 1:D-1){
    for (j in 1:p){
      beta[j, i] ~ normal(0., B_p); // uninformed prior
    }
  }
  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi);
    }
  }
}
