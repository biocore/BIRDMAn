data {
  int<lower=0> N;       // number of samples
  int<lower=0> D;       // number of dimensions
  real A;                                   // mean intercept
  int<lower=0> p;       // number of covariates
  vector[N] depth;      // sequencing depths of microbes
  matrix[N, p] x;       // covariate matrix
  array[N, D] int y;    // observed microbe abundances
  real<lower=0> B_p;    // stdev for intercept
  real<lower=0> phi_s;  // constant phi value
}

parameters {
  // parameters required for linear regression on the species means
  row_vector<offset=A, multiplier=B_p>[D-1] beta_0;
  matrix<multiplier=B_p>[p-1, D-1] beta_x;
  real inv_disp;
}

transformed parameters {
  matrix[p, D-1] beta_var = append_row(beta_0, beta_x);
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;

  lam = x*beta_var;
  for (n in 1:N){
    lam[n] += depth[n];
  }
  lam_clr = append_col(to_vector(rep_array(0, N)), lam);
}

model {
  inv_disp ~ lognormal(0, phi_s);

  for (i in 1:D-1){
    for (j in 1:p){
      beta_var[j, i] ~ normal(0., B_p); // uninformed prior
    }
  }

  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i], inv(inv_disp));
    }
  }
}
