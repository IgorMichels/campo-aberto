(() => {
  "use strict";

  const {
    fetchJSON,
    computeStrengthTiers,
    computeCard,
    renderStickerCard,
    escapeHTML,
    openStickerModal,
    openSharedStickerFromURLViaParams,
    siteURL,
  } = window.MatchesShared;
  const { displayTeamName } = window.CampoAberto;

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
      competitionName: document.getElementById("builder-home-competition-name"),
      competitionPrevBtn: document.getElementById("builder-home-competition-prev"),
      competitionNextBtn: document.getElementById("builder-home-competition-next"),
      crestPreview: document.getElementById("builder-home-crest"),
      nameLabel: document.getElementById("builder-home-name"),
      teamMenu: document.getElementById("builder-home-team-menu"),
      prevBtn: document.getElementById("builder-home-prev"),
      nextBtn: document.getElementById("builder-home-next"),
    },
    away: {
      competitionName: document.getElementById("builder-away-competition-name"),
      competitionPrevBtn: document.getElementById("builder-away-competition-prev"),
      competitionNextBtn: document.getElementById("builder-away-competition-next"),
      crestPreview: document.getElementById("builder-away-crest"),
      nameLabel: document.getElementById("builder-away-name"),
      teamMenu: document.getElementById("builder-away-team-menu"),
      prevBtn: document.getElementById("builder-away-prev"),
      nextBtn: document.getElementById("builder-away-next"),
    },
  };
  const otherSideOf = (side) => (side === "home" ? "away" : "home");
  const statusMessageEl = document.getElementById("status-message");

  function showStatus(message) {
    statusMessageEl.textContent = message || "";
    statusMessageEl.hidden = !message;
  }

  function findManifestCompetition(slug) {
    return state.manifest.competitions.find((c) => c.slug === slug);
  }

  function updateTeamPreview(side, teamName) {
    const roster = freePickRosters[side];
    const team = roster.find((t) => t.team === teamName);
    const { crestPreview, nameLabel } = builderSides[side];
    crestPreview.innerHTML = team
      ? `<img src="${siteURL(team.crest)}" alt="${escapeHTML(displayTeamName(team.team))}">`
      : "";
    nameLabel.textContent = team ? displayTeamName(team.team) : "";
  }

  function closeTeamMenu(side) {
    const { teamMenu, nameLabel } = builderSides[side];
    teamMenu.parentElement.classList.remove("is-open");
    nameLabel.setAttribute("aria-expanded", "false");
  }

  // Rebuilds a side's dropdown list from its current roster -- called after
  // every roster load/competition switch (new teams to list) and after
  // EITHER side's team changes (the OTHER side's list needs its disabled
  // option to track that change too, so both sides are always re-rendered
  // together, not just the one that moved).
  function renderTeamMenu(side) {
    const { teamMenu } = builderSides[side];
    const otherTeam = state.freePick[otherSideOf(side)].team;
    const currentTeam = state.freePick[side].team;

    teamMenu.innerHTML = freePickRosters[side]
      .map((team) => {
        const isMirror = team.team === otherTeam;
        return `<button
          type="button"
          data-team="${escapeHTML(team.team)}"
          ${team.team === currentTeam ? 'aria-current="true"' : ""}
          ${isMirror ? `disabled title="Já escolhido do outro lado"` : ""}
        >${escapeHTML(displayTeamName(team.team))}</button>`;
      })
      .join("");
  }

  // Sets a side's selected team and re-renders both the crest/name preview
  // and the resulting sticker -- the single path every default-selection,
  // arrow-cycle, and dropdown-pick codepath below goes through, so none of
  // them can ever fall out of sync with each other.
  function setFreePickTeam(side, teamName) {
    state.freePick[side].team = teamName;
    updateTeamPreview(side, teamName);
    renderBuilderResult();
    closeTeamMenu(side);
    renderTeamMenu("home");
    renderTeamMenu("away");
  }

  // Moves a side's selection by +-1 through its own roster (wrapping around
  // both ends), skipping over whatever team is currently selected on the
  // OTHER side -- the carousel never lands on a team-vs-itself matchup. Safe
  // against an infinite loop since at most one roster entry can equal the
  // other side's team name, so this always terminates within roster.length
  // steps; a single-team roster (nothing else to land on) is the only case
  // left unresolved, and simply keeps the current selection.
  function cycleTeam(side, direction) {
    const roster = freePickRosters[side];
    if (roster.length === 0) return;

    const otherTeam = state.freePick[otherSideOf(side)].team;
    let index = roster.findIndex((t) => t.team === state.freePick[side].team);
    if (index === -1) index = 0;

    for (let step = 0; step < roster.length; step++) {
      index = (index + direction + roster.length) % roster.length;
      if (roster.length === 1 || roster[index].team !== otherTeam) {
        setFreePickTeam(side, roster[index].team);
        return;
      }
    }
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

  // Moves a side's competition by +-1 through the manifest (wrapping around
  // both ends) -- the same arrow-cycled pattern as cycleTeam below, just one
  // level up. No self-matchup guard needed here: both sides defaulting to
  // the same competition is normal (cycleTeam already keeps the two TEAMS
  // distinct regardless of which competitions they come from).
  function cycleCompetition(side, direction) {
    const competitions = state.manifest.competitions;
    if (competitions.length === 0) return;

    let index = competitions.findIndex((c) => c.slug === state.freePick[side].slug);
    if (index === -1) index = 0;
    index = (index + direction + competitions.length) % competitions.length;

    const competition = competitions[index];
    const season = competition.seasons[competition.seasons.length - 1];
    loadFreePickRoster(side, competition.slug, season, state.freePick[otherSideOf(side)].team);
  }

  // Fetches the existing data/${slug}/${season}.json (already has every
  // team + crest + color, no new export needed for this), takes the roster
  // from its latest date's snapshot, and filters it to teams the model
  // actually has posterior-mean params for.
  //
  // Always lands on a real team (never a blank state) -- picks the first
  // roster entry, or the second one if the first would mirror avoidTeam
  // (the other side's current pick), so the builder never opens on a
  // team-vs-itself matchup. Awaited by callers one side at a time (never
  // both sides in parallel) specifically so this avoidTeam check always
  // sees the other side's already-settled selection, not a stale/blank one.
  async function loadFreePickRoster(side, slug, season, avoidTeam) {
    updateTeamPreview(side, null);
    state.freePick[side] = { slug, season, team: null };
    const competition = findManifestCompetition(slug);
    builderSides[side].competitionName.textContent = competition
      ? competition.competition.replace("Serie", "Série")
      : "";

    try {
      const data = await fetchJSON(`data/${slug}/${season}.json`);
      const lastDate = data.dates[data.dates.length - 1];
      const roster = data.snapshots[lastDate].teams
        .filter((team) => Boolean(state.params.teams[team.team]))
        .sort((a, b) => a.team.localeCompare(b.team, "pt-BR"));

      freePickRosters[side] = roster;
      if (roster.length === 0) {
        renderTeamMenu(side);
        renderBuilderResult();
        return;
      }
      const defaultTeam =
        roster.length > 1 && roster[0].team === avoidTeam ? roster[1].team : roster[0].team;
      setFreePickTeam(side, defaultTeam);
    } catch (error) {
      freePickRosters[side] = [];
      renderTeamMenu(side);
      renderBuilderResult();
    }
  }

  async function initFreePickBuilder() {
    if (!state.manifest.competitions || state.manifest.competitions.length === 0) return;

    const defaultCompetition = state.manifest.competitions[0];
    const defaultSeason = defaultCompetition.seasons[defaultCompetition.seasons.length - 1];

    // Sequential, not Promise.all: away's default pick needs to know home's
    // already-settled team to avoid mirroring it (see loadFreePickRoster).
    await loadFreePickRoster("home", defaultCompetition.slug, defaultSeason);
    await loadFreePickRoster(
      "away",
      defaultCompetition.slug,
      defaultSeason,
      state.freePick.home.team,
    );

    ["home", "away"].forEach((side) => {
      builderSides[side].competitionPrevBtn.addEventListener("click", () =>
        cycleCompetition(side, -1),
      );
      builderSides[side].competitionNextBtn.addEventListener("click", () =>
        cycleCompetition(side, 1),
      );

      builderSides[side].prevBtn.addEventListener("click", () => cycleTeam(side, -1));
      builderSides[side].nextBtn.addEventListener("click", () => cycleTeam(side, 1));

      const { nameLabel, teamMenu } = builderSides[side];
      nameLabel.addEventListener("click", () => {
        const isOpen = teamMenu.parentElement.classList.toggle("is-open");
        nameLabel.setAttribute("aria-expanded", String(isOpen));
        if (isOpen) closeTeamMenu(otherSideOf(side));
      });

      teamMenu.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-team]");
        if (button) setFreePickTeam(side, button.dataset.team);
      });
    });

    document.addEventListener("click", (event) => {
      ["home", "away"].forEach((side) => {
        if (!builderSides[side].teamMenu.parentElement.contains(event.target)) {
          closeTeamMenu(side);
        }
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeTeamMenu("home");
        closeTeamMenu("away");
      }
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
