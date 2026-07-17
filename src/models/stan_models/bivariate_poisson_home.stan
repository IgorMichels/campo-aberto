functions {
  real bivpois_log_prob(int x, int y, real lambda1, real lambda2, real lambda3) {
    int m = min(x, y);
    vector[m + 1] terms;

    for (k in 0:m) {
      terms[k + 1] = (x - k) * log(lambda1) - lgamma(x - k + 1) + (y - k) * log(lambda2) - lgamma(y - k + 1)
        + k * log(lambda3) - lgamma(k + 1);
    }

    return -(lambda1 + lambda2 + lambda3) + log_sum_exp(terms);
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
  vector[T] attack_raw_std;
  vector[T] defense_raw_std;
  real eta;
  real beta_home;
  real gamma_corr;
}

transformed parameters {
  vector[T] attack_raw = attack_raw_std;
  vector[T] defense_raw = defense_raw_std;
  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
  real lambda3 = exp(gamma_corr);
}

model {
  attack_raw_std ~ normal(0, 1);
  defense_raw_std ~ normal(0, 1);
  eta ~ normal(0, 1);
  beta_home ~ normal(0, 0.5);
  gamma_corr ~ normal(-2, 1);

  for (n in 1:N) {
    real lambda1 = exp(attack[team_i[n]] - defense[team_j[n]] + eta + beta_home);
    real lambda2 = exp(attack[team_j[n]] - defense[team_i[n]] + eta);

    target += game_weight[n] * bivpois_log_prob(y_i[n], y_j[n], lambda1, lambda2, lambda3);
  }
}

generated quantities {
  real log_lik = 0;

  for (n in 1:N) {
    real lambda1 = exp(attack[team_i[n]] - defense[team_j[n]] + eta + beta_home);
    real lambda2 = exp(attack[team_j[n]] - defense[team_i[n]] + eta);

    log_lik += game_weight[n] * bivpois_log_prob(y_i[n], y_j[n], lambda1, lambda2, lambda3);
  }
}
