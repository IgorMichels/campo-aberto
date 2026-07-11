(() => {
  "use strict";

  // Canonical team names are dict keys like "Flamengo / RJ" -- the " / UF"
  // suffix disambiguates clubs that share a base name across states (e.g.
  // "Botafogo / RJ" vs "Botafogo / SP", both real, both in data/assets/
  // club_infos.csv), so it stays on the data itself: params.json lookups,
  // search text, config guaranteed_slots, etc. all key off the full name.
  // This is the one shared place that strips it back off for anything
  // actually shown to a person -- every page (index/app.js, evolution.js,
  // matches_shared.js) calls this instead of hand-rolling its own
  // `name.split(" / ")[0]`, so "no UF visible anywhere on the site" only
  // has one implementation to keep correct.
  window.CampoAberto = window.CampoAberto || {};
  window.CampoAberto.displayTeamName = function displayTeamName(name) {
    return String(name ?? "").split(" / ")[0];
  };
})();
