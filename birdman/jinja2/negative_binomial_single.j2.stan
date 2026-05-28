data {
  int<lower=1> N;             // number of samples
  int<lower=1> p;             // number of covariates
  real A;                     // mean intercept
  vector[N] depth;        // log sequencing depths of microbes
  matrix[N, p] x;             // covariate matrix
  array[N] int y;             // observed microbe abundances

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
  real<offset=A, multiplier=B_p> beta_0;
  vector<multiplier=B_p>[p-1] beta_x;
  real<lower=0> inv_disp;

  // Random Effects
  {%- for re_name, num_factors in re_dict.items() %}
  vector[{{ num_factors }}] {{ re_name }}_eff;
  {%- endfor %}
  // End Random Effects
}

transformed parameters {
  vector[p] beta_var = append_row(beta_0, beta_x);
  vector[N] lam = x * beta_var + depth;

  // Random Effects
  for (n in 1:N) {
    {%- for re_name, num_factors in re_dict.items() %}
    lam[n] += {{ re_name }}_eff[{{ re_name }}_map[n]];
    {%- endfor %}
  }
  // End Random Effects
}

model {
  inv_disp ~ lognormal(0., inv_disp_sd);
  beta_0 ~ normal(A, B_p);
  beta_x ~ normal(0, B_p);

  // Random Effects
  {%- for re_name, num_factors in re_dict.items() %}
  {{ re_name }}_eff ~ normal(0, re_p);
  {%- endfor %}
  // End Random Effects

  y ~ neg_binomial_2_log(lam, inv(inv_disp));
}

generated quantities {
  vector[N] log_lhood;
  vector[N] y_predict;

  for (n in 1:N){
    y_predict[n] = neg_binomial_2_log_rng(lam[n], inv(inv_disp));
    log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | lam[n], inv(inv_disp));
  }
}
