(() => {
  "use strict";

  const {
    fetchJSON,
    buildTabs,
    updateTabSelection,
    buildSeasonSelect,
    computeStrengthTiers,
    computeCard,
    renderStickerCard,
    applyFilterToGrid,
    createPaginatedGrid,
    updateModelNote,
    openStickerModal,
    openSharedStickerFromURLViaParams,
  } = window.MatchesShared;

  const PAGE_SIZE = 10;

  const state = {
    matchesManifest: null,
    params: null,
    tiers: null,
    competitionSlug: null,
    season: null,
    filterText: "",
  };

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const modelNoteEl = document.getElementById("model-note");
  const teamFilterEl = document.getElementById("team-filter");
  const stickersGridEl = document.getElementById("stickers-grid");
  const loadMoreEl = document.getElementById("load-more");
  const statusMessageEl = document.getElementById("status-message");

  function showStatus(message) {
    statusMessageEl.textContent = message || "";
    statusMessageEl.hidden = !message;
  }

  const paginatedGrid = createPaginatedGrid({
    gridEl: stickersGridEl,
    moreButtonEl: loadMoreEl,
    pageSize: PAGE_SIZE,
    renderItem: (match) => renderStickerCard(match, state.tiers),
  });

  // A non-empty search bypasses pagination (every match is searched, not
  // just the currently-revealed page) -- reverts to the paginated view once
  // the query is cleared.
  function applyFilter() {
    if (state.filterText) {
      paginatedGrid.showAll();
    } else {
      paginatedGrid.reset();
    }
    applyFilterToGrid(stickersGridEl, state.filterText);
  }

  function findMatchesCompetition(slug) {
    return state.matchesManifest.competitions.find((c) => c.slug === slug);
  }

  // Loads every remaining not-yet-played fixture for one competition+season
  // (already sorted scheduled-first-by-date then postponed-last by the
  // export step -- no window/count cap any more, see
  // src.site.export_matches_data) and computes every card client-side via
  // computeCard. A card whose team has no entry in state.params.teams is
  // dropped with a console warning.
  async function loadMatches(slug, season) {
    showStatus("");
    try {
      const data = await fetchJSON(`data/${slug}/matches_${season}.json`);
      const cards = [];
      data.matches.forEach((base) => {
        const card = computeCard(base, state.params);
        if (card) {
          cards.push(card);
        } else {
          console.warn(
            `Sem parâmetros do modelo para ${base.home_team} x ${base.away_team}; jogo ignorado.`,
          );
        }
      });
      state.filterText = "";
      teamFilterEl.value = "";
      paginatedGrid.setItems(cards);
      applyFilterToGrid(stickersGridEl, "");
    } catch (error) {
      paginatedGrid.setItems([]);
      showStatus(`Não foi possível carregar os jogos: ${error.message}`);
    }
  }

  function selectCompetition(slug) {
    const competition = findMatchesCompetition(slug);
    if (!competition) return;

    state.competitionSlug = slug;
    updateTabSelection(tabsEl, slug);
    buildSeasonSelect(seasonSelectEl, competition.seasons);

    const season = competition.seasons.includes(state.season)
      ? state.season
      : competition.seasons[competition.seasons.length - 1];
    seasonSelectEl.value = String(season);
    state.season = season;

    loadMatches(state.competitionSlug, state.season);
  }

  seasonSelectEl.addEventListener("change", (event) => {
    state.season = Number(event.target.value);
    loadMatches(state.competitionSlug, state.season);
  });

  teamFilterEl.addEventListener("input", (event) => {
    state.filterText = event.target.value;
    applyFilter();
  });

  stickersGridEl.addEventListener("click", (event) => {
    const wrapper = event.target.closest(".sticker-wrapper");
    if (wrapper) {
      openStickerModal(wrapper, {
        home_slug: state.competitionSlug,
        home_season: String(state.season),
        away_slug: state.competitionSlug,
        away_season: String(state.season),
      });
    }
  });

  async function init() {
    try {
      const [matchesManifest, params] = await Promise.all([
        fetchJSON("data/matches_manifest.json"),
        fetchJSON("data/params.json"),
      ]);
      state.matchesManifest = matchesManifest;
      state.params = params;
      state.tiers = computeStrengthTiers(params.teams, params.model);
    } catch (error) {
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
      return;
    }

    updateModelNote(modelNoteEl, state.params.reference_date);

    if (!state.matchesManifest.competitions || state.matchesManifest.competitions.length === 0) {
      showStatus("Nenhum jogo disponível no momento.");
    } else {
      buildTabs(tabsEl, state.matchesManifest.competitions, selectCompetition);
      selectCompetition(state.matchesManifest.competitions[0].slug);
    }

    // Fire-and-forget: doesn't block the rest of init().
    openSharedStickerFromURLViaParams(state.params, state.tiers);
  }

  init();
})();
