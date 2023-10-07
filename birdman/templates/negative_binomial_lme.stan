data {
  int<lower=0> N;                           // number of sample IDs
  int<lower=0> S;                           // number of groups (subjects)
  int<lower=0> D;                           // number of dimensions
  real A;                                   // mean intercept
  int<lower=0> p;                           // number of covariates
  vector[N] depth;                          // log sequencing depths of microbes
  matrix[N, p] x;                           // covariate matrix
  array[N] int<lower=1, upper=S> subj_ids;  // mapping of samples to subject IDs
  array[N, D] int y;                        // observed microbe abundances

  real<lower=0> B_p;                        // stdev for covariate beta normal prior
  real<lower=0> inv_disp_sd;                // stdev for inv disp lognormal prior
  real<lower=0> u_p;                        // stdev for subject intercept normal prior
}

parameters {
  row_vector<offset=A, multiplier=B_p>[D-1] beta_0;
  matrix<multiplier=B_p>[p-1, D-1] beta_x;
  vector<lower=0>[D] inv_disp;
  matrix[S, D-1] subj_int;
}

transformed parameters {
  matrix[p, D-1] beta_var = append_row(beta_0, beta_x);
  matrix[N, D-1] lam;
  matrix[N, D] lam_clr;

  lam = x*beta_var;
  for (n in 1:N){
    lam[n] += subj_int[subj_ids[n]] + depth[n];
  }
  lam_clr = append_col(to_vector(rep_array(0, N)), lam);
}

model {
  inv_disp ~ lognormal(0, inv_disp_sd);

  for (i in 1:D-1){
    for (j in 1:p){
      beta_var[j, i] ~ normal(0., B_p); // uninformed prior
    }
    for (j in 1:S){
      subj_int[j, i] ~ normal(0., u_p);
    }
  }

  // generating counts
  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i], inv(inv_disp[i]));
    }
  }
}

generated quantities {
  array[N, D] int y_predict;
  array[N, D] real log_lhood;

  for (n in 1:N){
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(lam_clr[n, i], inv(inv_disp[i]));
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i], inv(inv_disp[i]));
    }
  }
}
