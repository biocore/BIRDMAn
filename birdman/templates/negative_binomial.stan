data {
  int<lower=0> N;             // number of samples
  int<lower=0> D;             // number of features
  real A;                     // mean intercept
  int<lower=0> p;             // number of covariates
  vector[N] depth;            // log sequencing depths
  matrix[N, p] x;             // covariate matrix
  array[N, D] int y;          // observed microbe abundances

  real<lower=0> B_p;          // stdev for beta normal prior
  real<lower=0> inv_disp_sd;  // stdev for inv disp lognormal prior
}

parameters {
  row_vector<offset=A, multiplier=B_p>[D-1] beta_0;
  matrix<multiplier=B_p>[p-1, D-1] beta_x;
  vector<lower=0>[D] inv_disp;
}

transformed parameters {
  matrix[p, D-1] beta_var = append_row(beta_0, beta_x);
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;

  lam_clr = append_col(to_vector(rep_array(0, N)), x*beta_var);
}

model {
  inv_disp ~ lognormal(0, inv_disp_sd);

  beta_0 ~ normal(A, B_p);
  for (i in 1:D-1){
    for (j in 1:p-1){
      beta_x[j, i] ~ normal(0., B_p);
    }
  }

  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i] + depth[n], inv(inv_disp[i]));
    }
  }
}

generated quantities {
  array[N, D] int y_predict;
  array[N, D] real log_lhood;

  for (n in 1:N){
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(lam_clr[n, i] + depth[n], inv(inv_disp[i]));
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i] + depth[n], inv(inv_disp[i]));
    }
  }
}
