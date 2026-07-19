(() => {
  "use strict";

  // Fills the 3 preview slots on the home page (one per home-card), each a
  // small "spoiler" of what that page actually shows -- not live/polling,
  // just rendered once from the same already-exported JSON every other page
  // reads. site/index.html lives at the site root (unlike matches/*.html),
  // so paths here are root-relative, no "../" prefix.
  //
  // Deliberately does NOT load matches_shared.js: that module wires up the
  // shared sticker modal at parse time (`document.getElementById("sticker-modal")`
  // + unconditional `.addEventListener` on it), which only exists on
  // matches/*.html -- loading it here throws before it ever assigns
  // `window.MatchesShared`. The only piece actually needed (total_strength,
  // the same measure that drives the sticker rarity border -- see
  // matches_shared.js::computeCard/computeStrengthTiers) is one line via
  // window.ScoreModels directly, so small local helpers below stand in for
  // the rest (escapeHTML/formatPercent/formatDateTimeLabel) instead.
  const { displayTeamName } = window.CampoAberto;

  // Broadcast-style acronym is exported per-team (see
  // src.site.export_site_data, sourced from data/assets/club_infos.csv's
  // hand-maintained "acronym" column) -- this falls back to the first 3
  // letters of the full name only for data exported before that field
  // existed, so a code deploy that lands before the next data regen still
  // renders something reasonable instead of "undefined".
  function acronymOf(team) {
    return team.acronym || displayTeamName(team.team).slice(0, 3).toUpperCase();
  }

  async function fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Falha ao carregar ${path}: HTTP ${response.status}`);
    }
    return response.json();
  }

  function escapeHTML(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatPercent(value) {
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  // Rounded to a whole percent -- the standings preview packs in enough
  // numeric columns (P/J/GP/SG plus 3 probabilities) that a decimal place on
  // every probability would overflow the card; exact figures are one click
  // away via "Ver mais".
  function formatPercentInt(value) {
    return `${Math.round((value || 0) * 100)}%`;
  }

  // Full ISO datetime -> "DD/MM HH:MM", local time -- same format as
  // matches_shared.js::formatDateTimeLabel.
  function formatDateTimeLabel(isoDatetime) {
    const date = new Date(isoDatetime);
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${day}/${month} ${hours}:${minutes}`;
  }

  // How many teams the Classificação preview shows -- past the top 4
  // (title-race podium), just filling out the card to roughly the same
  // height as the Jogos preview (3 matches) instead of leaving blank space.
  // Bumped to 8 then back down to 6 (2026-07-19): 8 plus the extra
  // Libertadores/Sul-Americana/GP/SG columns made every card taller than
  // Jogos actually needs, leaving a gap above "Ver mais" instead of closing
  // it -- 6 rows is the height that roughly matches Jogos' 3-match preview.
  const STANDINGS_PREVIEW_COUNT = 6;

  // Mini table (Classificação preview): rank, crest, acronym, points, games
  // played, goals for, goal diff, título/Libertadores/Sul-Americana
  // probabilities -- same default competition/season/date as app.js (first
  // manifest entry, latest season, latest snapshot). "P"/"J"/"GP"/"SG" match
  // the labels the full Classificação table already uses (app.js); wins
  // isn't shown since it isn't in the exported standings object (only
  // points, played, goals_for, goals_against, goal_diff) and can't be
  // derived from points+played alone. Returns the loaded {dates, snapshots}
  // plus the top-2 teams so renderEvolutionPreview can reuse this same fetch
  // instead of loading data/<slug>/<season>.json twice.
  async function renderStandingsPreview(containerEl) {
    const manifest = await fetchJSON("data/manifest.json");
    const competition = manifest.competitions[0];
    const season = competition.seasons[competition.seasons.length - 1];
    const data = await fetchJSON(`data/${competition.slug}/${season}.json`);
    const lastDate = data.dates[data.dates.length - 1];
    const topTeams = data.snapshots[lastDate].teams.slice(0, STANDINGS_PREVIEW_COUNT);

    const rows = topTeams
      .map(
        (team, index) => `
          <div class="preview-standings-row">
            <span class="preview-rank">${index + 1}</span>
            <span class="preview-team-cell">
              <img class="crest" src="${escapeHTML(team.crest)}" alt="" />
              <span class="preview-team" title="${escapeHTML(
                displayTeamName(team.team),
              )}">${escapeHTML(acronymOf(team))}</span>
            </span>
            <span class="preview-num">${team.standings.points}</span>
            <span class="preview-num">${team.standings.played}</span>
            <span class="preview-num">${team.standings.goals_for}</span>
            <span class="preview-num">${team.standings.goal_diff}</span>
            <span class="preview-num preview-value">${formatPercentInt(team.probs.title)}</span>
            <span class="preview-num preview-value">${formatPercentInt(
              team.probs.libertadores,
            )}</span>
            <span class="preview-num preview-value">${formatPercentInt(
              team.probs.sulamericana,
            )}</span>
          </div>`,
      )
      .join("");

    containerEl.innerHTML = `
      <div class="preview-standings">
        <div class="preview-standings-row preview-standings-head">
          <span></span>
          <span></span>
          <span class="preview-num-label" title="Pontos">P</span>
          <span class="preview-num-label" title="Jogos">J</span>
          <span class="preview-num-label" title="Gols pró">GP</span>
          <span class="preview-num-label" title="Saldo de gols">SG</span>
          <span class="preview-num-label" title="Probabilidade de título">Tít</span>
          <span class="preview-num-label" title="Probabilidade de Libertadores">Lib</span>
          <span class="preview-num-label" title="Probabilidade de Sul-Americana">Sula</span>
        </div>
        ${rows}
      </div>`;

    return { data, topTeams: topTeams.slice(0, 2) };
  }

  // Sparkline (Evolução preview): título probability only, for the current
  // leader and runner-up (reusing the standings fetch above -- no separate
  // data/<slug>/<season>.json load), against a fixed 0-100% y-axis so the
  // scale is legible on its own, not just relative to these two teams' range.
  function renderEvolutionPreview(containerEl, data, topTeams) {
    const series = topTeams.map((team) => ({
      name: team.team,
      color: team.color,
      acronym: acronymOf(team),
      values: data.dates.map((date) => {
        const found = data.snapshots[date].teams.find((t) => t.team === team.team);
        return found ? found.probs.title * 100 : null;
      }),
    }));

    if (data.dates.length < 2) {
      containerEl.textContent = "Histórico insuficiente ainda.";
      return;
    }

    const width = 220;
    const height = 128;
    const padLeft = 32;
    const padTop = 6;
    const padBottom = 8;
    const plotWidth = width - padLeft;
    const plotHeight = height - padTop - padBottom;
    const stepX = plotWidth / (data.dates.length - 1);
    const yFor = (value) => padTop + plotHeight - (value / 100) * plotHeight;

    const ticks = [0, 50, 100];
    const gridlines = ticks
      .map(
        (tick) => `
          <line x1="${padLeft}" y1="${yFor(tick).toFixed(1)}" x2="${width}" y2="${yFor(
            tick,
          ).toFixed(1)}"
            stroke="var(--gridline)" stroke-width="1" />
          <text x="${padLeft - 4}" y="${(yFor(tick) + 3).toFixed(1)}" text-anchor="end"
            font-size="9" fill="var(--text-muted)">${tick}%</text>`,
      )
      .join("");

    const lines = series
      .map((serie) => {
        const points = serie.values
          .map((value, index) => {
            if (value === null) return null;
            const x = padLeft + index * stepX;
            const y = yFor(value);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .filter((point) => point !== null)
          .join(" ");
        return `<polyline points="${points}" fill="none" stroke="${escapeHTML(
          serie.color,
        )}" stroke-width="2" />`;
      })
      .join("");

    const legend = series
      .map(
        (serie) => `
          <span class="preview-legend-item">
            <span class="preview-legend-swatch" style="background:${escapeHTML(
              serie.color,
            )}"></span>
            ${escapeHTML(serie.acronym)}
          </span>`,
      )
      .join("");

    containerEl.innerHTML = `
      <svg
        class="sparkline"
        viewBox="0 0 ${width} ${height}"
        preserveAspectRatio="none"
        role="img"
        aria-label="Evolução da probabilidade de título do líder e do vice"
      >
        ${gridlines}
        ${lines}
      </svg>
      <p class="preview-legend">${legend}</p>`;
  }

  function localDayKey(date) {
    return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
  }

  // Same "how strong are these two teams combined" measure that drives the
  // sticker rarity border on matches/*.html -- see
  // matches_shared.js::computeCard/computeStrengthTiers. Recomputed locally
  // (rather than via computeCard) since the home preview never renders a
  // scoreline grid, just needs this one number to rank candidates.
  function totalStrength(match, params) {
    const homeParams = params.teams[match.home_team];
    const awayParams = params.teams[match.away_team];
    if (!homeParams || !awayParams) return null;
    const model = window.ScoreModels[params.model];
    return model.teamStrength(homeParams) + model.teamStrength(awayParams);
  }

  function renderMatchItem(match) {
    return `
      <div class="preview-match">
        <div class="preview-match-row">
          <img class="crest" src="${escapeHTML(match.home_crest)}" alt="" />
          <span class="preview-match-team-name">${escapeHTML(
            displayTeamName(match.home_team),
          )}</span>
          <span class="preview-match-date">${escapeHTML(formatDateTimeLabel(match.date))}</span>
        </div>
        <div class="preview-match-row">
          <img class="crest" src="${escapeHTML(match.away_crest)}" alt="" />
          <span class="preview-match-team-name">${escapeHTML(
            displayTeamName(match.away_team),
          )}</span>
        </div>
      </div>`;
  }

  // Jogos preview -- the 3 most attractive upcoming matches (by
  // totalStrength above), ordered most-attractive-first. Candidates are
  // accumulated one calendar day at a time (today first, then following
  // days) until there are at least 3 to choose from, across every
  // competition/season with upcoming fixtures right now (Serie A and B are
  // fit jointly, so totalStrength is already comparable across both). Shows
  // "no games scheduled" once every remaining fixture is gone -- and hides
  // this card's "Ver mais" link in that case too, since there'd be nothing
  // left to see more of on matches/upcoming.html either.
  async function renderNextMatchesPreview(containerEl, moreLinkEl) {
    const [matchesManifest, params] = await Promise.all([
      fetchJSON("data/matches_manifest.json"),
      fetchJSON("data/params.json"),
    ]);

    const allMatches = [];
    for (const competition of matchesManifest.competitions || []) {
      for (const season of competition.seasons) {
        const data = await fetchJSON(`data/${competition.slug}/matches_${season}.json`);
        allMatches.push(...data.matches);
      }
    }

    const now = new Date();
    const upcoming = allMatches
      .filter((match) => match.status === "scheduled" && match.date && new Date(match.date) >= now)
      .sort((a, b) => new Date(a.date) - new Date(b.date));

    if (upcoming.length === 0) {
      containerEl.textContent = "Não há jogos programados no momento.";
      moreLinkEl.hidden = true;
      return;
    }

    const pool = [];
    let index = 0;
    while (index < upcoming.length && pool.length < 3) {
      const dayKey = localDayKey(new Date(upcoming[index].date));
      while (index < upcoming.length && localDayKey(new Date(upcoming[index].date)) === dayKey) {
        pool.push(upcoming[index]);
        index++;
      }
    }

    const ranked = pool
      .map((match) => ({ match, strength: totalStrength(match, params) }))
      .filter((entry) => entry.strength !== null)
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 3);

    if (ranked.length === 0) {
      containerEl.textContent = "Não há jogos programados no momento.";
      moreLinkEl.hidden = true;
      return;
    }

    containerEl.innerHTML = ranked.map((entry) => renderMatchItem(entry.match)).join("");
  }

  async function init() {
    const standingsEl = document.getElementById("preview-standings");
    const evolutionEl = document.getElementById("preview-evolution");
    const matchesEl = document.getElementById("preview-matches");
    const matchesMoreEl = document.getElementById("preview-matches-more");

    try {
      const { data, topTeams } = await renderStandingsPreview(standingsEl);
      renderEvolutionPreview(evolutionEl, data, topTeams);
    } catch (error) {
      standingsEl.textContent = "Não foi possível carregar a classificação.";
      evolutionEl.textContent = "Não foi possível carregar a evolução.";
    }

    try {
      await renderNextMatchesPreview(matchesEl, matchesMoreEl);
    } catch (error) {
      matchesEl.textContent = "Não foi possível carregar os próximos jogos.";
    }
  }

  init();
})();
