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
    escapeHTML,
    normalizeName,
    applyFilterToGrid,
    createPaginatedGrid,
    openStickerModal,
    openStickerModalForCard,
    siteURL,
  } = window.MatchesShared;

  const { displayTeamName } = window.CampoAberto;

  const PAGE_SIZE = 10;

  const state = {
    playedManifest: null,
    tiers: null,
    competitionSlug: null,
    season: null,
    filterText: "",
  };

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const teamFilterEl = document.getElementById("team-filter");
  const stickersGridEl = document.getElementById("stickers-grid");
  const loadMoreEl = document.getElementById("load-more");
  const statusMessageEl = document.getElementById("status-message");

  function showStatus(message) {
    statusMessageEl.textContent = message || "";
    statusMessageEl.hidden = !message;
  }

  // A played match with no prior model snapshot (has_model: false, see
  // src.site.export_matches_data._played_cards) gets this placeholder
  // instead of a probability grid -- deliberately a small sibling function
  // rather than forking renderStickerCard, since it shares almost nothing
  // with a real card (no heatmap, no win bar). `data-no-model="true"` marks
  // it so the click handler below skips opening the modal for it.
  function renderNoModelCard(match) {
    const searchText = normalizeName(`${match.home_team} ${match.away_team}`);
    return `
      <div class="sticker-wrapper"
        data-search="${escapeHTML(searchText)}"
        data-home-team="${escapeHTML(match.home_team)}"
        data-away-team="${escapeHTML(match.away_team)}"
        data-no-model="true">
        <div class="sticker-container no-model">
          <div class="sticker-header">
            <div class="team">
              <div class="sticker-crest"><img src="${siteURL(match.home_crest)}" alt="${escapeHTML(
                displayTeamName(match.home_team),
              )}"></div>
              <div class="team-name">${escapeHTML(displayTeamName(match.home_team))}</div>
            </div>
            <div class="score-center">
              <div class="real-score">${match.home_goals} - ${match.away_goals}</div>
            </div>
            <div class="team">
              <div class="sticker-crest"><img src="${siteURL(match.away_crest)}" alt="${escapeHTML(
                displayTeamName(match.away_team),
              )}"></div>
              <div class="team-name">${escapeHTML(displayTeamName(match.away_team))}</div>
            </div>
          </div>
          <p class="no-model-message">Sem modelo disponível ainda</p>
        </div>
      </div>
    `;
  }

  function renderPlayedCard(match) {
    if (!match.has_model) return renderNoModelCard(match);

    const card = computeCard(
      {
        home_team: match.home_team,
        away_team: match.away_team,
        home_crest: match.home_crest,
        away_crest: match.away_crest,
        home_color: match.home_color,
        away_color: match.away_color,
        date: match.date,
        final_score: { home: match.home_goals, away: match.away_goals },
      },
      match.params,
    );
    if (!card) return renderNoModelCard(match);
    return renderStickerCard(card, state.tiers);
  }

  const paginatedGrid = createPaginatedGrid({
    gridEl: stickersGridEl,
    moreButtonEl: loadMoreEl,
    pageSize: PAGE_SIZE,
    renderItem: renderPlayedCard,
  });

  function applyFilter() {
    if (state.filterText) {
      paginatedGrid.showAll();
    } else {
      paginatedGrid.reset();
    }
    applyFilterToGrid(stickersGridEl, state.filterText);
  }

  function findPlayedCompetition(slug) {
    return state.playedManifest.competitions.find((c) => c.slug === slug);
  }

  // Loads every played match for one competition+season (already
  // most-recent-first, each with its own embedded params slice, by the
  // export step -- see src.site.export_matches_data._played_cards). The tier
  // scale (state.tiers) is recomputed per season from the union of every
  // match's own embedded params, so the rarity border stays meaningful
  // across a whole page even though each card's total_strength came from a
  // different historical snapshot.
  async function loadPlayed(slug, season) {
    showStatus("");
    try {
      const data = await fetchJSON(`data/${slug}/played_${season}.json`);
      const paramsTeams = {};
      data.matches.forEach((match) => {
        if (match.has_model) Object.assign(paramsTeams, match.params.teams);
      });
      state.tiers = computeStrengthTiers(paramsTeams);

      state.filterText = "";
      teamFilterEl.value = "";
      paginatedGrid.setItems(data.matches);
      applyFilterToGrid(stickersGridEl, "");
    } catch (error) {
      paginatedGrid.setItems([]);
      showStatus(`Não foi possível carregar os jogos passados: ${error.message}`);
    }
  }

  function selectCompetition(slug) {
    const competition = findPlayedCompetition(slug);
    if (!competition) return;

    state.competitionSlug = slug;
    updateTabSelection(tabsEl, slug);
    buildSeasonSelect(seasonSelectEl, competition.seasons);

    const season = competition.seasons.includes(state.season)
      ? state.season
      : competition.seasons[competition.seasons.length - 1];
    seasonSelectEl.value = String(season);
    state.season = season;

    loadPlayed(state.competitionSlug, state.season);
  }

  seasonSelectEl.addEventListener("change", (event) => {
    state.season = Number(event.target.value);
    loadPlayed(state.competitionSlug, state.season);
  });

  teamFilterEl.addEventListener("input", (event) => {
    state.filterText = event.target.value;
    applyFilter();
  });

  stickersGridEl.addEventListener("click", (event) => {
    const wrapper = event.target.closest(".sticker-wrapper");
    if (!wrapper || wrapper.dataset.noModel === "true") return;
    openStickerModal(wrapper, {
      slug: state.competitionSlug,
      season: String(state.season),
    });
  });

  // Counterpart to matches_shared.js's openSharedStickerFromURLViaParams,
  // but for the played-match shape (?slug&season&home_team&away_team) --
  // looks the match back up in its own played_<season>.json rather than
  // recomputing from current params, since a played card's probabilities
  // came from a specific historical snapshot, not "whatever's current".
  async function openSharedStickerFromURLForPlayed() {
    const searchParams = new URLSearchParams(location.search);
    const slug = searchParams.get("slug");
    const season = searchParams.get("season");
    const homeTeam = searchParams.get("home_team");
    const awayTeam = searchParams.get("away_team");
    if (!slug || !season || !homeTeam || !awayTeam) return;

    try {
      const data = await fetchJSON(`data/${slug}/played_${season}.json`);
      const match = data.matches.find((m) => m.home_team === homeTeam && m.away_team === awayTeam);
      if (!match) {
        console.warn("Link compartilhado aponta para um jogo que não existe mais.");
        return;
      }
      if (!match.has_model) {
        console.warn("Link compartilhado aponta para um jogo sem modelo disponível.");
        return;
      }

      // A tier scale computed from just this one match's own embedded
      // params (the shared-link visitor hasn't loaded this competition's
      // full season yet, so there's no wider distribution to compare
      // against) -- good enough for a single restored card's border.
      const tiers = computeStrengthTiers(match.params.teams);
      const card = computeCard(
        {
          home_team: match.home_team,
          away_team: match.away_team,
          home_crest: match.home_crest,
          away_crest: match.away_crest,
          home_color: match.home_color,
          away_color: match.away_color,
          date: match.date,
          final_score: { home: match.home_goals, away: match.away_goals },
        },
        match.params,
      );
      if (!card) {
        console.warn("Link compartilhado aponta para um jogo sem parâmetros do modelo.");
        return;
      }

      openStickerModalForCard(card, tiers, { slug, season: String(season) });
    } catch (error) {
      console.warn(`Não foi possível abrir a figurinha compartilhada: ${error.message}`);
    }
  }

  async function init() {
    try {
      state.playedManifest = await fetchJSON("data/played_manifest.json");
    } catch (error) {
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
      return;
    }

    if (!state.playedManifest.competitions || state.playedManifest.competitions.length === 0) {
      showStatus("Nenhum jogo passado disponível no momento.");
    } else {
      buildTabs(tabsEl, state.playedManifest.competitions, selectCompetition);
      selectCompetition(state.playedManifest.competitions[0].slug);
    }

    openSharedStickerFromURLForPlayed();
  }

  init();
})();
