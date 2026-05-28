data {
  int<lower=1> N;             // number of samples
  int<lower=1> D;             // number of features
  int<lower=1> p;             // number of covariates
  real A;                     // mean intercept
  vector[N] depth;            // log sequencing depths of microbes
  matrix[N, p] x;             // covariate matrix
  array[N, D] int y;          // observed microbe abundances

  real<lower=0> B_p;          // stdev for beta normal prior
  real<lower=0> inv_disp_sd;  // stdev for inv disp lognormal prior

  // Random Effects
  real<lower=0> re_p;         // stdev for random effect normal prior
  {% for re_name, num_factors in re_dict.items() %}
  array[N] int<lower=1, upper={{ num_factors }}> {{ re_name }}_map;
  {%- endfor %}
  // End Random Effects
}

parameters {
  row_vector<offset=A, multiplier=B_p>[D-1] beta_0;
  matrix<multiplier=B_p>[p-1, D-1] beta_x;
  vector<lower=0>[D] inv_disp;

  // Random Effects
  {%- for re_name, num_factors in re_dict.items() %}
  matrix[{{ num_factors }}, D-1] {{ re_name }}_eff;
  {%- endfor %}
  // End Random Effects
}

transformed parameters {
  matrix[p, D-1] beta_var = append_row(beta_0, beta_x);
  matrix[N, D-1] lam = x * beta_var;
  matrix[N, D] lam_clr;

  // Random Effects
  for (n in 1:N) {
    lam[n] += depth[n];
    {%- for re_name, num_factors in re_dict.items() %}
    lam[n] += {{ re_name }}_eff[{{ re_name }}_map[n]];
    {%- endfor %}
  }
  // End Random Effects

  lam_clr = append_col(to_vector(rep_array(0, N)), lam);
}

model {
  inv_disp ~ lognormal(0., inv_disp_sd);

  beta_0 ~ normal(A, B_p);
  for (i in 1:D-1){
    for (j in 1:p-1){
      beta_x[j, i] ~ normal(0., B_p);
    }
    // Random Effects
    {%- for re_name, num_factors in re_dict.items() %}
    for (j in 1:{{ num_factors }}) {
      {{ re_name }}_eff[j, i] ~ normal(0, re_p);
    }
    {%- endfor %}
    // End Random Effects
  }

  for (n in 1:N){
    for (i in 1:D){
      target += neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i], inv_disp[i]);
    }
  }
}

generated quantities {
  array[N, D] int y_predict;
  array[N, D] real log_lhood;

  for (n in 1:N){
    for (i in 1:D){
      y_predict[n, i] = neg_binomial_2_log_rng(lam_clr[n, i], inv_disp[i]);
      log_lhood[n, i] = neg_binomial_2_log_lpmf(y[n, i] | lam_clr[n, i], inv_disp[i]);
    }
  }
}
