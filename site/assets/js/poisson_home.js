(() => {
  "use strict";

  // The "poisson_home" entry in window.ScoreModels (see score_models.js) --
  // today's production model, and the ONLY implementation of this specific
  // math in the whole project (see plans/confrontos_rework.md, Step 3).
  // Every parameter this needs (posterior-mean attack/defense per team +
  // shared eta/beta_home/rho) is already shipped to the browser in
  // site/data/params.json, so there is no Python port: a second
  // implementation would be pure duplication.
  //
  // This is the exact closed-form evaluation of the correction
  // src/models/poisson_home.stan's dc_log_prob already fits and
  // src/models/adapters/poisson_home.py already codes as a rejection-
  // sampling accept/reject bound -- this file is the first place this repo
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

  // Exact Poisson pmf for k = 0..maxK, no tail bucketing -- unlike
  // poissonPmfWithTail, every entry is the literal P(X = k) term (used where
  // an individual scoreline's exact probability is needed, e.g. a played
  // match's real result, rather than a bucketed display grid).
  function poissonPmfArray(mu, maxK) {
    const probs = new Array(maxK + 1);
    let p = Math.exp(-mu);
    for (let k = 0; k <= maxK; k++) {
      probs[k] = p;
      p = (p * mu) / (k + 1);
    }
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

  // Per-match scoring rates from each side's own team params (this model's
  // shape: {attack, defense}) plus the shared params (this model's shape:
  // {eta, beta_home, rho}) -- object arguments, not positional scalars, so
  // window.ScoreModels' contract (see score_models.js) doesn't assume every
  // model has exactly this parameter set.
  function matchRates(homeTeamParams, awayTeamParams, sharedParams) {
    const { attack: attackHome, defense: defenseHome } = homeTeamParams;
    const { attack: attackAway, defense: defenseAway } = awayTeamParams;
    const { eta, beta_home: betaHome } = sharedParams;
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
  // home_win/draw/away_win from that full, unbucketed grid, then slices out
  // its own top-left (maxGoals+1) x (maxGoals+1) corner for display -- each
  // display cell is the EXACT probability of that single scoreline (since
  // maxGoals=4 is well inside outcomeCap=10, no tail-bucketing touches it),
  // not a summed "N or more" bucket, so it doesn't sum to 1: the remaining
  // mass (5+ goals on either side) simply isn't shown on the grid.
  //
  // `best` (the single most-likely EXACT scoreline) is picked from this same
  // fine grid -- not by scanning `grid`, since a scoreline outside the
  // displayed 0..maxGoals corner (rare, but possible for a very lopsided
  // matchup) could still be the true most-likely one.
  function scorelineProbabilities(muHome, muAway, sharedParams, maxGoals = 4, outcomeCap = 10) {
    const { rho } = sharedParams;
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
    let best = { home: 0, away: 0, prob: -1 };
    for (let x = 0; x <= outcomeCap; x++) {
      for (let y = 0; y <= outcomeCap; y++) {
        const p = fine[x][y] / total;
        if (x > y) homeWin += p;
        else if (x === y) draw += p;
        else awayWin += p;
        if (p > best.prob) best = { home: x, away: y, prob: p };
      }
    }

    const grid = [];
    for (let x = 0; x <= maxGoals; x++) {
      const row = new Array(maxGoals + 1);
      for (let y = 0; y <= maxGoals; y++) {
        row[y] = fine[x][y] / total;
      }
      grid.push(row);
    }

    return { grid, home_win: homeWin, draw, away_win: awayWin, best };
  }

  // Exact probability of one specific scoreline, for any (home, away) --
  // unlike scorelineProbabilities' `grid`, which buckets everything at
  // maxGoals=4 or more into a single ">= 4" cell (needed there so the
  // heatmap stays a fixed 5x5), this is used for a played match's real
  // result (matches_shared.js::computeCard's final_score_prob), which can
  // legitimately be any scoreline, including blowouts past 4 goals.
  // `outcomeCap` is widened to cover (home, away) if either exceeds the
  // default -- the joint grid it renormalizes against must include the
  // queried cell itself.
  function scorelineProbabilityAt(muHome, muAway, sharedParams, home, away, outcomeCap = 10) {
    const { rho } = sharedParams;
    const cap = Math.max(outcomeCap, home, away);
    const pHome = poissonPmfArray(muHome, cap);
    const pAway = poissonPmfArray(muAway, cap);

    let total = 0;
    for (let x = 0; x <= cap; x++) {
      for (let y = 0; y <= cap; y++) {
        total += dixonColesTau(x, y, muHome, muAway, rho) * pHome[x] * pAway[y];
      }
    }

    const raw = dixonColesTau(home, away, muHome, muAway, rho) * pHome[home] * pAway[away];
    return raw / total;
  }

  // Symmetric "how strong is this team" scalar, used only to derive the
  // sticker card's rarity border (see matches_shared.js::computeStrengthTiers/
  // strengthTierClass) -- not part of the scoreline math itself.
  function teamStrength(teamParams) {
    return teamParams.attack + teamParams.defense;
  }

  window.ScoreModels = window.ScoreModels || {};
  window.ScoreModels.poisson_home = {
    matchRates,
    scorelineProbabilities,
    scorelineProbabilityAt,
    teamStrength,
  };
})();
