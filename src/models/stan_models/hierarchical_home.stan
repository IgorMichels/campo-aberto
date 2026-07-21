data {
  int<lower=1> N;
  int<lower=1> T;
  array[N] int<lower=1, upper=T> team_i; // home team
  array[N] int<lower=1, upper=T> team_j; // away team
  array[N] int<lower=0> y_i; // home goals
  array[N] int<lower=0> y_j; // away goals
  vector[N] game_weight;
  array[T] int<lower=1, upper=4> group; // 1=ficou-A 2=elevador-A-B 3=ficou-B 4=elevador-B-C
  // Weak informative prior (default sd=1, wide) reflecting the a priori
  // strength order ficou-A > elevador-A-B > ficou-B > elevador-B-C -- only
  // shifts each group's MEAN; the data is free to invert this order.
  vector[4] group_prior_mean;
  real<lower=0> group_prior_sd;
}

parameters {
  vector[T] attack_raw_std;
  vector[T] defense_raw_std;
  vector[4] mu_attack;
  vector[4] mu_defense;
  real eta;
  real beta_home;
}

transformed parameters {
  // Every team's spread is fixed at 1 (attack_raw_std/defense_raw_std ~
  // normal(0, 1) below), same as poisson_home_no_rho -- only the group MEAN
  // is hierarchical here. An earlier version also estimated a per-group
  // sigma_attack/sigma_defense (vector<lower=0>[4] ~ normal(0, 1)), but that
  // over-shrank the small groups (elevador-A-B, elevador-B-C have few teams
  // per checkpoint, so their sigma posterior was noisy and prior-dominated,
  // pulled toward 0) and backtested worse than poisson_home. Fixing the
  // scale at 1 and keeping only the group-mean shift matched or beat
  // poisson_home's pooled Brier (2022-2026 walk-forward backtest).
  vector[T] attack_raw = mu_attack[group] + attack_raw_std;
  vector[T] defense_raw = mu_defense[group] + defense_raw_std;
  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
}

model {
  attack_raw_std ~ normal(0, 1);
  defense_raw_std ~ normal(0, 1);
  mu_attack ~ normal(group_prior_mean, group_prior_sd);
  mu_defense ~ normal(group_prior_mean, group_prior_sd);
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
