(() => {
  "use strict";

  const state = {
    manifest: null, // data/manifest.json -- every competition/season with a roster (drives the free-pick builder)
    matchesManifest: null, // data/matches_manifest.json -- only competitions/seasons with real upcoming cards
    params: null, // data/params.json -- shared eta/beta_home/rho + per-team posterior-mean attack/defense
    strengthTiers: null, // {prime, mid, legendaryTeams} -- total_strength percentile cutoffs (Prime/Mid) + top-3-team set (Prime Icon Moments), computed once in init()
    competitionSlug: null,
    season: null,
    cardsData: null, // {matches: [...]} computed cards for the currently loaded competition+season
    filterText: "",
    freePick: {
      home: { slug: null, season: null, team: null },
      away: { slug: null, season: null, team: null },
    },
  };

  // Free-pick rosters (team/crest/color, filtered to teams with known
  // params), keyed by side -- kept out of `state.freePick` on purpose: the
  // plan's own state shape for free-pick is just {slug, season, team} per
  // side, this is derived/cached data, not user selection.
  const freePickRosters = { home: [], away: [] };

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const modelNoteEl = document.getElementById("model-note");
  const teamFilterEl = document.getElementById("team-filter");
  const stickersGridEl = document.getElementById("stickers-grid");
  const scrollLeftEl = document.getElementById("scroll-left");
  const scrollRightEl = document.getElementById("scroll-right");
  const statusMessageEl = document.getElementById("status-message");
  const modalEl = document.getElementById("sticker-modal");
  const modalOverlayEl = document.getElementById("sticker-modal-overlay");
  const modalContentEl = document.getElementById("sticker-modal-content");

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

  async function fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Falha ao carregar ${path}: HTTP ${response.status}`);
    }
    return response.json();
  }

  function showStatus(message) {
    statusMessageEl.textContent = message || "";
    statusMessageEl.hidden = !message;
  }

  function escapeHTML(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function normalizeName(value) {
    return String(value ?? "")
      .trim()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase();
  }

  function formatPercent(value) {
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  // A handful of real club colors (data/assets/club_infos.csv's
  // primary_color column) are literally #000000 -- a legitimate real color
  // identity (Corinthians, Botafogo/RJ, Santos/SP, Vasco da Gama/RJ,
  // Atlético Mineiro/MG and others are genuinely black-and-white clubs), not
  // a data bug. Using that color directly as *text* color on the sticker's
  // own near-black background (#111 container, further darkened by the
  // footer's rgba(0,0,0,0.3) overlay) makes the text completely invisible.
  // readableTextColor keeps the real color for text whenever it's legible
  // and substitutes a readable fallback (white) only when it isn't --
  // callers must keep using the real, unmodified home_color/away_color
  // everywhere else (the win bar's background fill, .sticker-bg-blur's
  // wash), so a team's actual color identity is untouched except for this
  // one text-legibility fix.
  //
  // Threshold picked against the REAL exported color distribution, not
  // guessed: computed relative luminance for every primary_color in
  // data/assets/club_infos.csv -- every real club color's luminance is
  // either exactly 0 (the ~11 clubs whose real color IS #000000) or
  // >= 0.0174 (Remo/PA, the next-darkest real club color). Any threshold
  // strictly between those two values isolates exactly the true #000000
  // cases without touching any other real, legitimately-dark-but-still-
  // visible club color -- in particular Flamengo (#C52613, luminance
  // 0.133) and Palmeiras (#006437, luminance 0.094), both already
  // confirmed rendering correctly with their own real color in Step 8, stay
  // comfortably above this threshold and untouched.
  const READABLE_TEXT_LUMINANCE_THRESHOLD = 0.015;

  function hexToRgb(hex) {
    const match = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? "").trim());
    if (!match) return null;
    const value = parseInt(match[1], 16);
    return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
  }

  // Standard WCAG/sRGB relative luminance formula.
  function relativeLuminance({ r, g, b }) {
    const channel = (c) => {
      const s = c / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  }

  function isUnreadablyDark(hex) {
    const rgb = hexToRgb(hex);
    return !!rgb && relativeLuminance(rgb) < READABLE_TEXT_LUMINANCE_THRESHOLD;
  }

  function readableTextColor(hex) {
    return isUnreadablyDark(hex) ? "#ffffff" : hex;
  }

  // Bare "YYYY-MM-DD" -> "DD/MM/AAAA" -- still needed for the model note
  // (params.reference_date has no time component), even though card dates
  // now carry a full ISO datetime (see formatDateTimeLabel below).
  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
  }

  // Full ISO datetime -> "DD/MM HH:MM", local time -- used for real fixture
  // cards (matches_<season>.json's "date" field is a UTC instant; the
  // browser's own timezone conversion via `Date` is exactly what we want
  // here, no manual offset math).
  function formatDateTimeLabel(isoDatetime) {
    const date = new Date(isoDatetime);
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${day}/${month} ${hours}:${minutes}`;
  }

  function findMatchesCompetition(slug) {
    return state.matchesManifest.competitions.find((c) => c.slug === slug);
  }

  function findManifestCompetition(slug) {
    return state.manifest.competitions.find((c) => c.slug === slug);
  }

  function buildTabs(competitions) {
    tabsEl.innerHTML = "";
    competitions.forEach((competition) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tab-button";
      button.textContent = competition.competition.replace("Serie", "Série");
      button.setAttribute("role", "tab");
      button.dataset.slug = competition.slug;
      button.addEventListener("click", () => selectCompetition(competition.slug));
      tabsEl.appendChild(button);
    });
  }

  function updateTabSelection() {
    Array.from(tabsEl.children).forEach((button) => {
      const selected = button.dataset.slug === state.competitionSlug;
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
  }

  function buildSeasonSelect(seasons) {
    seasonSelectEl.innerHTML = "";
    seasons.forEach((season) => {
      const option = document.createElement("option");
      option.value = String(season);
      option.textContent = String(season);
      seasonSelectEl.appendChild(option);
    });
  }

  function buildCompetitionOptions(selectEl, competitions) {
    selectEl.innerHTML = "";
    competitions.forEach((competition) => {
      const option = document.createElement("option");
      option.value = competition.slug;
      option.textContent = competition.competition.replace("Serie", "Série");
      selectEl.appendChild(option);
    });
  }

  // Finds the single highest-probability scoreline in a match's 5x5 grid.
  function bestScore(scores) {
    let best = { home: 0, away: 0, prob: -1 };
    Object.entries(scores).forEach(([key, prob]) => {
      if (prob > best.prob) {
        const [home, away] = key.split("_").map(Number);
        best = { home, away, prob };
      }
    });
    return best;
  }

  function renderHeatmap(scores, best) {
    const maxProb = Math.max(...Object.values(scores), 0.01);
    let html = "";

    for (let h = 4; h >= 0; h--) {
      for (let a = 0; a <= 4; a++) {
        const prob = scores[`${h}_${a}`] || 0;
        const isBest = h === best.home && a === best.away;
        const alpha = Math.min(1, (prob / maxProb) * 0.7 + 0.1);
        const bg = isBest
          ? "rgba(255, 255, 255, 0.95)"
          : prob === 0
            ? "rgba(255, 255, 255, 0.04)"
            : `rgba(255, 255, 255, ${alpha})`;
        const probText = prob < 0.001 ? "0%" : formatPercent(prob);

        html += `
          <div class="heatmap-cell ${isBest ? "best" : ""}" style="background: ${bg}">
            <div class="prob">${probText}</div>
            <div class="score">${h}x${a}</div>
          </div>
        `;
      }
    }

    return html;
  }

  // What the sticker's top-info bar shows: a scheduled real fixture shows
  // its date/time, a postponed one shows "Data a definir", a free-pick card
  // (no `status` field at all) shows nothing.
  function topInfoText(match) {
    if (match.status === "scheduled") return match.date ? formatDateTimeLabel(match.date) : "";
    if (match.status === "postponed") return "Data a definir";
    return "";
  }

  // Renders one match's probability "sticker": crests, most-likely score,
  // the 25-cell scoreline heatmap and the home/draw/away win-probability bar.
  function renderStickerCard(match) {
    const best = bestScore(match.scores);
    const heatmapHTML = renderHeatmap(match.scores, best);

    const borderClass = strengthTierClass(match);

    const searchText = normalizeName(`${match.home_team} ${match.away_team}`);
    const topInfo = topInfoText(match);

    return `
      <div class="sticker-wrapper"
        data-search="${escapeHTML(searchText)}">
        <div class="sticker-container ${borderClass}" style="--home-color: ${
          match.home_color
        }; --away-color: ${match.away_color};">
          <div class="sticker-bg-blur"></div>

          <div class="sticker-glass">
            <div class="sticker-top-info">
              ${topInfo ? `<span>${escapeHTML(topInfo)}</span>` : ""}
            </div>

            <div class="sticker-header">
              <div class="team">
                <div class="sticker-crest"><img src="${match.home_crest}" alt="${escapeHTML(
                  match.home_team,
                )}"></div>
                <div class="team-name">${escapeHTML(match.home_team)}</div>
              </div>

              <div class="score-center">
                <div class="most-likely">${best.home} - ${best.away}</div>
                <div class="most-likely-prob">${formatPercent(best.prob)}</div>
              </div>

              <div class="team">
                <div class="sticker-crest"><img src="${match.away_crest}" alt="${escapeHTML(
                  match.away_team,
                )}"></div>
                <div class="team-name">${escapeHTML(match.away_team)}</div>
              </div>
            </div>

            <div class="heatmap-wrapper">
              <div class="heatmap-grid">${heatmapHTML}</div>
            </div>

            <div class="sticker-footer">
              <div class="footer-bar-container">
                <div class="f-bar home" style="width: ${
                  match.home_win * 100
                }%; background: ${readableTextColor(match.home_color)};"></div>
                <div class="f-bar draw" style="width: ${match.draw * 100}%"></div>
                <div class="f-bar away" style="width: ${
                  match.away_win * 100
                }%; background: ${readableTextColor(match.away_color)};"></div>
              </div>
              <div class="footer-stats-row">
                <div class="f-stat home" style="color: ${readableTextColor(
                  match.home_color,
                )};">${formatPercent(match.home_win)}</div>
                <div class="f-stat draw">${formatPercent(match.draw)}</div>
                <div class="f-stat away" style="color: ${readableTextColor(
                  match.away_color,
                )};">${formatPercent(match.away_win)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // The single shared computation for BOTH real-fixture cards and free-pick
  // cards -- see plans/confrontos_rework.md Step 5. `base` is either a real
  // fixture read off matches_<season>.json (has date/status) or a free-pick
  // selection (no date/status). Looks up each side's posterior-mean
  // attack/defense in state.params.teams, runs the Dixon-Coles math
  // (window.DixonColes, Step 3) entirely client-side, and returns the full
  // card object renderStickerCard expects. Returns null (caller
  // skips/warns) if either team has no known params.
  function computeCard(base) {
    const homeParams = state.params.teams[base.home_team];
    const awayParams = state.params.teams[base.away_team];
    if (!homeParams || !awayParams) return null;

    const { muHome, muAway } = DixonColes.matchRates(
      homeParams.attack,
      homeParams.defense,
      awayParams.attack,
      awayParams.defense,
      state.params.eta,
      state.params.beta_home,
    );
    const { grid, home_win, draw, away_win } = DixonColes.scorelineProbabilities(
      muHome,
      muAway,
      state.params.rho,
    );

    const scores = {};
    grid.forEach((row, home) => {
      row.forEach((prob, away) => {
        scores[`${home}_${away}`] = prob;
      });
    });

    // Symmetric in home/away (a plain sum) -- measures "how strong are
    // these two teams combined", independent of who hosts. Drives the
    // card's rarity border (see computeStrengthTiers/strengthTierClass).
    const total_strength =
      homeParams.attack + homeParams.defense + awayParams.attack + awayParams.defense;

    return { ...base, scores, home_win, draw, away_win, total_strength };
  }

  // Rarity-tier cutoffs for the card border, derived from the *current*
  // data's own distribution of total_strength rather than hardcoded --
  // Step 7 of plans/confrontos_rework.md demonstrated this scale can shift
  // by ~5x whenever the Stan model's priors/data change, so a hardcoded
  // cutoff would silently stop meaning anything (or bucket every card into
  // one tier) the next time the model is refit. Enumerates total_strength
  // across every unordered pair of teams with known params (symmetric, so
  // home/away order doesn't matter -- no need for all ordered pairs) and
  // returns percentile cutoffs: the top ~15% of matchups by combined
  // strength get the "Prime" tier, the next ~25% get the "Mid" tier, the
  // rest render with the "Base" tier (still the bottom 60%, most common --
  // just its own styled class now, not a plain/no-class default). Computed
  // once in init(), right after state.params loads, and cached on
  // state.strengthTiers. Tier NAMES/colors evoke FIFA Ultimate Team's Icon
  // card tiers (see the "Rarity restyle" section of
  // plans/confrontos_rework.md); the underlying P60/P85 percentile logic
  // computed here is unchanged by that restyle -- only which CSS class each
  // cutoff maps to (see strengthTierClass) changed.
  //
  // Also ranks individual teams by their own team_strength (attack +
  // defense, same two values summed for one team instead of two) and
  // records the top 3 team names as `legendaryTeams` -- a categorical (not
  // percentile-based) 4th tier, rarer than the top 15% "Prime" tier: a
  // matchup is "Prime Icon Moments" iff BOTH its home and away team are
  // among these top 3, checked before falling through to the percentile
  // lookup (see strengthTierClass). This is a separate, sequential check,
  // not folded into the percentile computation above -- the top-15%/25%/60%
  // split is computed exactly as before, unperturbed by which teams happen
  // to qualify for the top-3 rule.
  function computeStrengthTiers(paramsTeams) {
    const teamNames = Object.keys(paramsTeams);
    const totals = [];
    for (let i = 0; i < teamNames.length; i++) {
      const a = paramsTeams[teamNames[i]];
      for (let j = i + 1; j < teamNames.length; j++) {
        const b = paramsTeams[teamNames[j]];
        totals.push(a.attack + a.defense + b.attack + b.defense);
      }
    }
    totals.sort((x, y) => x - y);

    const percentile = (p) => {
      if (totals.length === 0) return Infinity;
      const idx = Math.min(totals.length - 1, Math.floor(p * (totals.length - 1)));
      return totals[idx];
    };

    const legendaryTeams = new Set(
      teamNames
        .map((name) => ({
          name,
          team_strength: paramsTeams[name].attack + paramsTeams[name].defense,
        }))
        .sort((x, y) => y.team_strength - x.team_strength)
        .slice(0, 3)
        .map((t) => t.name),
    );

    return {
      prime: percentile(0.85), // top ~15% of matchups by combined strength ("Prime")
      mid: percentile(0.6), // next ~25% ("Mid")
      legendaryTeams, // top-3 individually-strongest teams ("Prime Icon Moments" tier)
    };
  }

  // Maps a card to its border rarity class using the tiers computed once by
  // computeStrengthTiers. Checks the categorical "Prime Icon Moments" rule
  // first (both home_team and away_team among the top-3 strongest teams) --
  // regardless of which two of the three, and regardless of home/away
  // order -- and only falls through to the ordinary total_strength
  // percentile lookup (Prime/Mid/Base) if it doesn't qualify. Falls back to
  // the "Base" tier if tiers haven't been computed yet (shouldn't happen in
  // practice -- init() computes them before any card is rendered) or if
  // total_strength is somehow missing.
  function strengthTierClass(match) {
    const tiers = state.strengthTiers;
    if (!tiers) return "border-base";

    if (
      tiers.legendaryTeams &&
      tiers.legendaryTeams.has(match.home_team) &&
      tiers.legendaryTeams.has(match.away_team)
    ) {
      return "border-icon-moments";
    }

    const totalStrength = match.total_strength;
    if (totalStrength == null) return "border-base";
    if (totalStrength >= tiers.prime) return "border-prime";
    if (totalStrength >= tiers.mid) return "border-mid";
    return "border-base";
  }

  // Renders every card once; the search box then only toggles visibility
  // (applyFilter) instead of re-rendering, so cards keep their scroll
  // position while the user types.
  function renderAllStickers() {
    const matches = state.cardsData ? state.cardsData.matches : [];
    stickersGridEl.innerHTML = matches.map(renderStickerCard).join("");
    applyFilter();
  }

  function applyFilter() {
    const query = normalizeName(state.filterText);
    stickersGridEl.querySelectorAll(".sticker-wrapper").forEach((wrapper) => {
      const matches = !query || wrapper.dataset.search.includes(query);
      wrapper.style.display = matches ? "" : "none";
    });
  }

  function openStickerModal(wrapperEl) {
    const container = wrapperEl.querySelector(".sticker-container");
    if (!container) return;
    modalContentEl.innerHTML = container.outerHTML;
    modalEl.classList.add("active");
  }

  function closeStickerModal() {
    modalEl.classList.remove("active");
    modalContentEl.innerHTML = "";
  }

  stickersGridEl.addEventListener("click", (event) => {
    const wrapper = event.target.closest(".sticker-wrapper");
    if (wrapper) openStickerModal(wrapper);
  });

  builderResultEl.addEventListener("click", (event) => {
    const wrapper = event.target.closest(".sticker-wrapper");
    if (wrapper) openStickerModal(wrapper);
  });

  modalOverlayEl.addEventListener("click", closeStickerModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeStickerModal();
  });

  teamFilterEl.addEventListener("input", (event) => {
    state.filterText = event.target.value;
    applyFilter();
  });

  scrollLeftEl.addEventListener("click", () => {
    stickersGridEl.scrollBy({ left: -300, behavior: "smooth" });
  });

  scrollRightEl.addEventListener("click", () => {
    stickersGridEl.scrollBy({ left: 300, behavior: "smooth" });
  });

  // Loads real, not-yet-played fixtures for one competition+season (already
  // sorted scheduled-first-by-date then postponed-last by the export step)
  // and computes every card client-side via computeCard. A card whose team
  // has no entry in state.params.teams is dropped with a console warning
  // (export_matches_data.py already tries to avoid this at export time, but
  // params.json can legitimately lag behind a newer matches_<season>.json).
  async function loadMatches(slug, season) {
    showStatus("");
    stickersGridEl.innerHTML = "";
    try {
      const data = await fetchJSON(`data/${slug}/matches_${season}.json`);
      const cards = [];
      data.matches.forEach((base) => {
        const card = computeCard(base);
        if (card) {
          cards.push(card);
        } else {
          console.warn(
            `Sem parâmetros do modelo para ${base.home_team} x ${base.away_team}; confronto ignorado.`,
          );
        }
      });
      state.cardsData = { matches: cards };
      state.filterText = "";
      teamFilterEl.value = "";
      renderAllStickers();
    } catch (error) {
      state.cardsData = null;
      showStatus(`Não foi possível carregar os confrontos: ${error.message}`);
    }
  }

  function selectCompetition(slug) {
    const competition = findMatchesCompetition(slug);
    if (!competition) return;

    state.competitionSlug = slug;
    updateTabSelection();
    buildSeasonSelect(competition.seasons);

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

  // ── Free-pick builder ──
  // FIFA-style: pick a competition, then a team, for each side independently
  // (any two teams, any competition, including cross-division). Re-renders
  // its own result area on every change -- no submit button, the
  // computation is cheap pure JS (same computeCard() as real fixtures).

  function updateCrestPreview(side, teamName) {
    const roster = freePickRosters[side];
    const team = roster.find((t) => t.team === teamName);
    const previewEl = builderSides[side].crestPreview;
    previewEl.innerHTML = team ? `<img src="${team.crest}" alt="${escapeHTML(team.team)}">` : "";
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

    const card = computeCard({
      home_team: homeTeam.team,
      away_team: awayTeam.team,
      home_crest: homeTeam.crest,
      away_crest: awayTeam.crest,
      home_color: homeTeam.color,
      away_color: awayTeam.color,
    });

    builderResultEl.innerHTML = card
      ? renderStickerCard(card)
      : '<p class="status-message">Sem parâmetros do modelo suficientes para este confronto.</p>';
  }

  // Fetches the existing data/${slug}/${season}.json (already has every
  // team + crest + color, no new export needed for this), takes the roster
  // from its latest date's snapshot, and filters it to teams the model
  // actually has posterior-mean params for (state.params.teams) -- a team
  // from an older season spelling wouldn't be there and couldn't be
  // computed anyway.
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

  function updateModelNote() {
    if (!state.params || !state.params.reference_date) {
      modelNoteEl.textContent = "";
      return;
    }
    modelNoteEl.textContent = `Baseado no modelo ajustado até ${formatDateLabel(
      state.params.reference_date,
    )}.`;
  }

  async function init() {
    try {
      const [manifest, matchesManifest, params] = await Promise.all([
        fetchJSON("data/manifest.json"),
        fetchJSON("data/matches_manifest.json"),
        fetchJSON("data/params.json"),
      ]);
      state.manifest = manifest;
      state.matchesManifest = matchesManifest;
      state.params = params;
      state.strengthTiers = computeStrengthTiers(params.teams);
    } catch (error) {
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
      return;
    }

    updateModelNote();

    if (!state.matchesManifest.competitions || state.matchesManifest.competitions.length === 0) {
      showStatus("Nenhum confronto disponível no momento.");
    } else {
      buildTabs(state.matchesManifest.competitions);
      selectCompetition(state.matchesManifest.competitions[0].slug);
    }

    initFreePickBuilder();
  }

  init();
})();
