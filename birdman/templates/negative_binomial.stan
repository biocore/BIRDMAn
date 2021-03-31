data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  int<lower=0> p;       // number of covariates
  real depth[N];        // log sequencing depths of microbes
  matrix[N, p] x;       // covariate matrix
  int y[N, D];          // observed microbe abundances
  real<lower=0> B_p;    // stdev for Beta Normal prior
  real<lower=0> phi_s;  // scale for dispersion Cauchy prior
}

parameters {
  // parameters required for linear regression on the species means
  matrix[p, D-1] beta;
  vector<lower=0>[D] reciprocal_phi;
}

transformed parameters {
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;
  vector<lower=0>[D] phi;

  for (i in 1:D){
    phi[i] = 1. / reciprocal_phi[i];
  }

  lam = x * beta;
  lam_clr = append_col(to_vector(rep_array(0, N)), lam);
}

model {
  // setting priors ...
  for (i in 1:D){
    reciprocal_phi[i] ~ cauchy(0., phi_s);
  }
  for (i in 1:D-1){
    for (j in 1:p){
      beta[j, i] ~ normal(0., B_p); // uninformed prior
    }
  }
  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi[i]);
    }
  }
}

generated quantities {
  matrix[N, D] y_predict;
  matrix[N, D] log_lhood;

  for (n in 1:N){
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(depth[n] + lam_clr[n, i], phi[i]);
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | depth[n] + lam_clr[n, i], phi[i]);
    }
  }
}
