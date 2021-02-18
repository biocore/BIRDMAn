# Separate priors for intercept and host_common_name
data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  int<lower=0> p;       // number of covariates
  real depth[N];        // sequencing depths of microbes
  matrix[N, p] x;       // covariate matrix
  int y[N, D];          // observed microbe abundances
  real<lower=0> B_p_1;  // stdev for Beta Normal prior
  real<lower=0> B_p_2;  // stdev for Beta Normal prior
  real<lower=0> phi_s;  // constant phi value
}

parameters {
  // parameters required for linear regression on the species means
  matrix[p, D-1] beta_var;
}

transformed parameters {
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;
  vector[N] z;

  z = to_vector(rep_array(0, N));
  lam = x * beta_var;
  lam_clr = append_col(z, lam);
}

model {
  // setting priors ...
  for (i in 1:D-1){
    beta_var[1, i] ~ normal(0., B_p_1); // uninformed prior
    beta_var[2, i] ~ normal(0., B_p_2); // uninformed prior
  }
  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi_s);
    }
  }
}

