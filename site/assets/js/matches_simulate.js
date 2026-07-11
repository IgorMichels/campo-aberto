(() => {
  "use strict";

  const {
    fetchJSON,
    buildCompetitionOptions,
    computeStrengthTiers,
    computeCard,
    renderStickerCard,
    escapeHTML,
    openStickerModal,
    openSharedStickerFromURLViaParams,
    siteURL,
  } = window.MatchesShared;

  const state = {
    manifest: null, // data/manifest.json -- every competition/season with a roster
    params: null, // data/params.json -- shared eta/beta_home/rho + per-team posterior-mean attack/defense
    tiers: null,
    freePick: {
      home: { slug: null, season: null, team: null },
      away: { slug: null, season: null, team: null },
    },
  };

  // Derived/cached roster data (team/crest/color per side), kept out of
  // state.freePick on purpose -- that's just {slug, season, team}, this is
  // the fetched roster the dropdowns are populated from.
  const freePickRosters = { home: [], away: [] };

  const builderResultEl = document.getElementById("builder-result");
  const builderSides = {
    home: {
      competitionSelect: document.getElementById("builder-home-competition"),
      teamSelect: document.getElementById("builder-home-team"),
      crestPreview: document.getElementById("builder-home-crest"),
    },
    away: {
      competitionSelect: document.getElementById("builder-away-competition"),
      teamSelect: document.getElementById("builder-away-team"),
      crestPreview: document.getElementById("builder-away-crest"),
    },
  };
  const statusMessageEl = document.getElementById("status-message");

  function showStatus(message) {
    statusMessageEl.textContent = message || "";
    statusMessageEl.hidden = !message;
  }

  function findManifestCompetition(slug) {
    return state.manifest.competitions.find((c) => c.slug === slug);
  }

  function updateCrestPreview(side, teamName) {
    const roster = freePickRosters[side];
    const team = roster.find((t) => t.team === teamName);
    const previewEl = builderSides[side].crestPreview;
    previewEl.innerHTML = team
      ? `<img src="${siteURL(team.crest)}" alt="${escapeHTML(team.team)}">`
      : "";
  }

  function renderBuilderResult() {
    const home = state.freePick.home;
    const away = state.freePick.away;
    if (!home.team || !away.team) {
      builderResultEl.innerHTML = "";
      return;
    }

    const homeTeam = freePickRosters.home.find((t) => t.team === home.team);
    const awayTeam = freePickRosters.away.find((t) => t.team === away.team);
    if (!homeTeam || !awayTeam) {
      builderResultEl.innerHTML = "";
      return;
    }

    const card = computeCard(
      {
        home_team: homeTeam.team,
        away_team: awayTeam.team,
        home_crest: homeTeam.crest,
        away_crest: awayTeam.crest,
        home_color: homeTeam.color,
        away_color: awayTeam.color,
      },
      state.params,
    );

    builderResultEl.innerHTML = card
      ? renderStickerCard(card, state.tiers)
      : '<p class="status-message">Sem parâmetros do modelo suficientes para este confronto.</p>';
  }

  // Fetches the existing data/${slug}/${season}.json (already has every
  // team + crest + color, no new export needed for this), takes the roster
  // from its latest date's snapshot, and filters it to teams the model
  // actually has posterior-mean params for.
  async function loadFreePickRoster(side, slug, season) {
    const { teamSelect } = builderSides[side];
    teamSelect.innerHTML = '<option value="">Carregando...</option>';
    updateCrestPreview(side, null);
    state.freePick[side] = { slug, season, team: null };

    try {
      const data = await fetchJSON(`data/${slug}/${season}.json`);
      const lastDate = data.dates[data.dates.length - 1];
      const roster = data.snapshots[lastDate].teams
        .filter((team) => Boolean(state.params.teams[team.team]))
        .sort((a, b) => a.team.localeCompare(b.team, "pt-BR"));

      freePickRosters[side] = roster;
      teamSelect.innerHTML =
        '<option value="">Selecione um time...</option>' +
        roster
          .map(
            (team) => `<option value="${escapeHTML(team.team)}">${escapeHTML(team.team)}</option>`,
          )
          .join("");
      renderBuilderResult();
    } catch (error) {
      freePickRosters[side] = [];
      teamSelect.innerHTML = '<option value="">Erro ao carregar times</option>';
      renderBuilderResult();
    }
  }

  function initFreePickBuilder() {
    if (!state.manifest.competitions || state.manifest.competitions.length === 0) return;

    ["home", "away"].forEach((side) => {
      buildCompetitionOptions(builderSides[side].competitionSelect, state.manifest.competitions);
    });

    const defaultCompetition = state.manifest.competitions[0];
    const defaultSeason = defaultCompetition.seasons[defaultCompetition.seasons.length - 1];

    ["home", "away"].forEach((side) => {
      builderSides[side].competitionSelect.value = defaultCompetition.slug;
      loadFreePickRoster(side, defaultCompetition.slug, defaultSeason);

      builderSides[side].competitionSelect.addEventListener("change", (event) => {
        const competition = findManifestCompetition(event.target.value);
        if (!competition) return;
        const season = competition.seasons[competition.seasons.length - 1];
        loadFreePickRoster(side, competition.slug, season);
      });

      builderSides[side].teamSelect.addEventListener("change", (event) => {
        const teamName = event.target.value || null;
        state.freePick[side].team = teamName;
        updateCrestPreview(side, teamName);
        renderBuilderResult();
      });
    });
  }

  builderResultEl.addEventListener("click", (event) => {
    const wrapper = event.target.closest(".sticker-wrapper");
    if (wrapper) {
      openStickerModal(wrapper, {
        home_slug: state.freePick.home.slug,
        home_season: String(state.freePick.home.season),
        away_slug: state.freePick.away.slug,
        away_season: String(state.freePick.away.season),
      });
    }
  });

  async function init() {
    try {
      const [manifest, params] = await Promise.all([
        fetchJSON("data/manifest.json"),
        fetchJSON("data/params.json"),
      ]);
      state.manifest = manifest;
      state.params = params;
      state.tiers = computeStrengthTiers(params.teams);
    } catch (error) {
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
      return;
    }

    initFreePickBuilder();

    // Fire-and-forget: doesn't block the rest of init().
    openSharedStickerFromURLViaParams(state.params, state.tiers);
  }

  init();
})();
