(() => {
  "use strict";

  // Populates the shared footer's "last updated" line from data/params.json's
  // reference_date -- the single globally-latest reference date across
  // competitions (see site/README.md's "Data lineage" section for why this
  // file is the canonical "as of" source, not a per-competition `dates`
  // array in data/<slug>/<season>.json, which can lag on a competition with
  // an older last-played match). Shared across all 6 pages
  // (index/probabilities/evolution/matches/*.html) via one
  // <p id="last-updated"> in each page's <footer>, wired up the same way
  // nav.js shares the nav dropdown across every page instead of duplicating
  // fetch+format logic per page.

  // This script's own <script src="..."> attribute already encodes how deep
  // the current page sits below site/ (e.g. "assets/js/last_updated.js" from
  // site root, "../assets/js/last_updated.js" from matches/*.html) -- reused
  // here instead of hardcoding a fixed depth the way matches_shared.js's
  // SITE_ROOT constant does for its own (always one-level-down) pages.
  const scriptEl = document.currentScript;
  const SITE_ROOT = scriptEl.getAttribute("src").replace(/assets\/js\/.*$/, "");

  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
  }

  async function init() {
    const el = document.getElementById("last-updated");
    if (!el) return;

    try {
      const response = await fetch(`${SITE_ROOT}data/params.json`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const params = await response.json();
      el.textContent = `Dados atualizados em ${formatDateLabel(params.reference_date)}`;
    } catch (error) {
      // Leave the line blank rather than showing a broken date -- the rest
      // of the footer (repo link) still renders fine either way.
      console.warn(`Não foi possível carregar a data de atualização: ${error.message}`);
    }
  }

  init();
})();
