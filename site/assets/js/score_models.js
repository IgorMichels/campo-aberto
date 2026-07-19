(() => {
  "use strict";

  // Registry of client-side scoreline models. Each entry implements the
  // same 3-function contract:
  //   matchRates(homeTeamParams, awayTeamParams, sharedParams) -> {muHome, muAway}
  //   scorelineProbabilities(muHome, muAway, sharedParams, maxGoals, outcomeCap)
  //     -> {grid, home_win, draw, away_win}
  //   scorelineProbabilityAt(muHome, muAway, sharedParams, home, away, outcomeCap)
  //     -> probability (exact, for any single (home, away) scoreline, not just
  //        ones inside the bucketed grid's 0..maxGoals range)
  //   teamStrength(teamParams) -> number (drives the sticker card's rarity border)
  //
  // matches_shared.js::computeCard/computeStrengthTiers dispatch through
  // `window.ScoreModels[params.model]` (params.model comes straight from
  // site/data/params.json or a played card's own embedded snapshot -- see
  // src.site.export_matches_data) instead of calling one hardcoded
  // implementation. Adding a candidate model (see src/models/registry.py's
  // Python-side counterpart) is one new JS file registering itself here,
  // plus a <script> tag alongside poisson_home.js's on every page that
  // needs it -- nothing else in this file, or in matches_shared.js, changes.
  window.ScoreModels = window.ScoreModels || {};
})();
