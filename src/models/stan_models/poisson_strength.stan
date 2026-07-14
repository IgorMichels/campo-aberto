functions {
  real dc_log_prob(int y_i, int y_j, real mu, real lambda, real rho) {
    real log_prob = poisson_lpmf(y_i | mu) + poisson_lpmf(y_j | lambda);

    if (y_i == 0 && y_j == 0) {
      log_prob += log(fmax(0.001, 1 - mu * lambda * rho));
    } else if (y_i == 0 && y_j == 1) {
      log_prob += log(fmax(0.001, 1 + mu * rho));
    } else if (y_i == 1 && y_j == 0) {
      log_prob += log(fmax(0.001, 1 + lambda * rho));
    } else if (y_i == 1 && y_j == 1) {
      log_prob += log(fmax(0.001, 1 - rho));
    }

    return log_prob;
  }
}

data {
  int<lower=1> N;
  int<lower=1> T;
  array[N] int<lower=1, upper=T> team_i; // home team
  array[N] int<lower=1, upper=T> team_j; // away team
  array[N] int<lower=0> y_i; // home goals
  array[N] int<lower=0> y_j; // away goals
  vector[N] game_weight;
}

parameters {
  vector[T] strength_raw_std;
  real eta;
  real beta_home;
  real<lower=-0.5, upper=0.5> rho;
}

transformed parameters {
  vector[T] strength_raw = strength_raw_std;
  vector[T] strength = strength_raw - mean(strength_raw);
}

model {
  strength_raw_std ~ normal(0, 1);
  eta ~ normal(0, 1);
  beta_home ~ normal(0, 0.5);
  rho ~ normal(0, 0.1);

  for (n in 1:N) {
    real mu = exp(strength[team_i[n]] - strength[team_j[n]] + eta + beta_home);
    real lambda = exp(strength[team_j[n]] - strength[team_i[n]] + eta);

    target += game_weight[n] * dc_log_prob(y_i[n], y_j[n], mu, lambda, rho);
  }
}

generated quantities {
  real log_lik = 0;

  for (n in 1:N) {
    real mu = exp(strength[team_i[n]] - strength[team_j[n]] + eta + beta_home);
    real lambda = exp(strength[team_j[n]] - strength[team_i[n]] + eta);

    log_lik += game_weight[n] * dc_log_prob(y_i[n], y_j[n], mu, lambda, rho);
  }
}
