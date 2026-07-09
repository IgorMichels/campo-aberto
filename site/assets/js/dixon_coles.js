(() => {
  "use strict";

  // Dixon-Coles-adjusted independent-Poisson scoreline probabilities -- the
  // ONLY implementation of this math in the whole project (see
  // plans/confrontos_rework.md, Step 3). Every parameter this needs
  // (posterior-mean attack/defense per team + shared eta/beta_home/rho) is
  // already shipped to the browser in site/data/params.json, so there is no
  // Python port: a second implementation would be pure duplication.
  //
  // This is the exact closed-form evaluation of the correction
  // src/models/poisson_home.stan's dc_log_prob already fits and
  // src/simulation/simulate.py::simulate_scores already codes as
  // tau00/tau01/tau10/tau11 -- there, only ever used as a rejection-sampling
  // accept/reject bound. dixon_coles.js is the first place this repo
  // evaluates it as an explicit closed-form probability.
  //
  //   mu_home = exp(attack_home - defense_away + eta + beta_home)
  //   mu_away = exp(attack_away - defense_home + eta)
  //   tau(0,0) = max(0, 1 - mu_home*mu_away*rho)
  //   tau(1,0) = max(0, 1 + mu_away*rho)
  //   tau(0,1) = max(0, 1 + mu_home*rho)
  //   tau(1,1) = max(0, 1 - rho)
  //   tau(x,y) = 1 otherwise
  //   P(x,y) = tau(x,y) * Poisson(x; mu_home) * Poisson(y; mu_away)

  // Poisson pmf for k = 0..maxExact-1, plus the tail P(X >= maxExact) as the
  // last entry (so the returned array always sums to exactly 1, regardless of
  // mu or maxExact -- computing the tail as a complement rather than summing
  // further Poisson terms avoids losing precision for a large mu).
  function poissonPmfWithTail(mu, maxExact) {
    const probs = new Array(maxExact + 1).fill(0);
    let p = Math.exp(-mu);
    let cumulative = 0;
    for (let k = 0; k < maxExact; k++) {
      probs[k] = p;
      cumulative += p;
      p = (p * mu) / (k + 1);
    }
    probs[maxExact] = Math.max(0, 1 - cumulative);
    return probs;
  }

  // The Dixon-Coles low-score correction: only the four (x, y) in {0, 1}^2
  // cells are reweighted; every other cell keeps its independent-Poisson mass.
  function dixonColesTau(x, y, muHome, muAway, rho) {
    if (x === 0 && y === 0) return Math.max(0, 1 - muHome * muAway * rho);
    if (x === 0 && y === 1) return Math.max(0, 1 + muHome * rho);
    if (x === 1 && y === 0) return Math.max(0, 1 + muAway * rho);
    if (x === 1 && y === 1) return Math.max(0, 1 - rho);
    return 1;
  }

  // Per-match scoring rates from each side's posterior-mean attack/defense
  // plus the shared eta/beta_home. Note the parameter order: attackHome,
  // defenseHome, attackAway, defenseAway (each side's own attack/defense
  // together), not grouped by "home rate inputs" / "away rate inputs".
  function matchRates(attackHome, defenseHome, attackAway, defenseAway, eta, betaHome) {
    const muHome = Math.exp(attackHome - defenseAway + eta + betaHome);
    const muAway = Math.exp(attackAway - defenseHome + eta);
    return { muHome, muAway };
  }

  // Full scoreline distribution for one match.
  //
  // Computes a fine (outcomeCap+1) x (outcomeCap+1) joint grid (independent
  // Poisson marginals with the Dixon-Coles tau correction on its 4 special
  // cells, last row/column of each marginal being the ">= outcomeCap" tail),
  // renormalizes it by its own sum (the tau correction and the tail
  // truncation both nudge the raw total away from exactly 1), derives
  // home_win/draw/away_win from that full, unbucketed grid, then buckets it
  // down to a (maxGoals+1) x (maxGoals+1) grid for display (each of the last
  // row/column/corner is the sum of every fine cell with that many goals or
  // more, matching the fine grid's own tail convention).
  function scorelineProbabilities(muHome, muAway, rho, maxGoals = 4, outcomeCap = 10) {
    const pHome = poissonPmfWithTail(muHome, outcomeCap);
    const pAway = poissonPmfWithTail(muAway, outcomeCap);

    const fine = [];
    for (let x = 0; x <= outcomeCap; x++) {
      const row = new Array(outcomeCap + 1);
      for (let y = 0; y <= outcomeCap; y++) {
        row[y] = dixonColesTau(x, y, muHome, muAway, rho) * pHome[x] * pAway[y];
      }
      fine.push(row);
    }

    let total = 0;
    for (let x = 0; x <= outcomeCap; x++) {
      for (let y = 0; y <= outcomeCap; y++) {
        total += fine[x][y];
      }
    }

    let homeWin = 0;
    let draw = 0;
    let awayWin = 0;
    for (let x = 0; x <= outcomeCap; x++) {
      for (let y = 0; y <= outcomeCap; y++) {
        const p = fine[x][y] / total;
        if (x > y) homeWin += p;
        else if (x === y) draw += p;
        else awayWin += p;
      }
    }

    const grid = [];
    for (let x = 0; x <= maxGoals; x++) {
      const row = new Array(maxGoals + 1).fill(0);
      const xs = x === maxGoals ? range(x, outcomeCap) : [x];
      for (let y = 0; y <= maxGoals; y++) {
        const ys = y === maxGoals ? range(y, outcomeCap) : [y];
        let cell = 0;
        for (const xi of xs) {
          for (const yi of ys) {
            cell += fine[xi][yi];
          }
        }
        row[y] = cell / total;
      }
      grid.push(row);
    }

    return { grid, home_win: homeWin, draw, away_win: awayWin };
  }

  // Inclusive integer range [start, end].
  function range(start, end) {
    const values = [];
    for (let i = start; i <= end; i++) values.push(i);
    return values;
  }

  window.DixonColes = { matchRates, scorelineProbabilities };
})();
