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
  vector[T] attack_raw_std;
  vector[T] defense_raw_std;
  real eta;
  real beta_home;
}

transformed parameters {
  vector[T] attack_raw = attack_raw_std;
  vector[T] defense_raw = defense_raw_std;
  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
}

model {
  attack_raw_std ~ normal(0, 1);
  defense_raw_std ~ normal(0, 1);
  eta ~ normal(0, 1);
  beta_home ~ normal(0, 0.5);

  for (n in 1:N) {
    real mu = exp(attack[team_i[n]] - defense[team_j[n]] + eta + beta_home);
    real lambda = exp(attack[team_j[n]] - defense[team_i[n]] + eta);

    target += game_weight[n] * (poisson_lpmf(y_i[n] | mu) + poisson_lpmf(y_j[n] | lambda));
  }
}

generated quantities {
  real log_lik = 0;

  for (n in 1:N) {
    real mu = exp(attack[team_i[n]] - defense[team_j[n]] + eta + beta_home);
    real lambda = exp(attack[team_j[n]] - defense[team_i[n]] + eta);

    log_lik += game_weight[n] * (poisson_lpmf(y_i[n] | mu) + poisson_lpmf(y_j[n] | lambda));
  }
}
