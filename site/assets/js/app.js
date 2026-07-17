(() => {
  "use strict";

  // Shared with evolution.js/matches_shared.js via team_display.js, loaded before
  // this file -- see that file for why club names keep their " / UF" state
  // suffix in the data (disambiguates clubs like "Botafogo / RJ" vs
  // "Botafogo / SP") even though it's never shown to the user.
  const { displayTeamName } = window.CampoAberto;

  const state = {
    manifest: null,
    competitionSlug: null,
    season: null,
    date: null,
    seasonData: null, // {columns, dates, snapshots} for the currently loaded competition+season
    sort: null, // {key, kind, direction} once the user clicks a header; null = default classificação order
  };

  // Real (not simulated) standings, always the same 4 columns regardless of
  // competition -- unlike data.columns (probabilities), this isn't config-driven,
  // so it's simplest as a fixed constant here rather than round-tripping it
  // through the export script's JSON. Reuses the exact same {key, label,
  // children} shape leafColumns()/renderTableHead() already understand, so no
  // separate rendering path is needed for the group header.
  const STANDINGS_GROUP = {
    key: "standings",
    label: "Classificação",
    children: [
      { key: "points", label: "P", kind: "int" },
      { key: "played", label: "J", kind: "int" },
      { key: "goals_for", label: "GP", kind: "int" },
      { key: "goals_against", label: "GC", kind: "int" },
      { key: "goal_diff", label: "SG", kind: "int" },
    ],
  };

  // Not part of data.columns (it's the crest+name cell, not a probability/stat),
  // but sortable the same way -- alphabetically by team name.
  const TEAM_COLUMN = { key: "team", label: "Time", kind: "team" };

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const dateSelectEl = document.getElementById("date-select");
  const tableHeadRowEl = document.getElementById("table-head-row");
  const tableBodyEl = document.getElementById("table-body");
  const statusMessageEl = document.getElementById("status-message");
  const zoneLegendEl = document.getElementById("zone-legend");

  // Every zone src.site.export_site_data's _real_classification can produce
  // that actually gets a color (see style.css's tr.zone-* rules) -- best
  // outcome first. "title"/aggregate-total columns are deliberately absent:
  // they never surface as a team's own `zone` (title is always subsumed by
  // the broader spot it's nested in) or aren't a real position-based zone at
  // all (an aggregate's own "Geral" total). Labels are looked up from the
  // season's own `columns` tree, so the legend only shows entries that
  // actually apply to the competition currently on screen (e.g. Serie A
  // never shows "Acesso" rows, Serie B never shows "Libertadores" ones).
  const ZONE_LEGEND_ORDER = [
    "libertadores_grupos",
    "libertadores_pre",
    "sulamericana",
    "direct_promotion",
    "playoff_promotion",
    "rebaixamento",
  ];

  async function fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Falha ao carregar ${path}: HTTP ${response.status}`);
    }
    return response.json();
  }

  function showStatus(message) {
    if (message) {
      statusMessageEl.textContent = message;
      statusMessageEl.hidden = false;
    } else {
      statusMessageEl.textContent = "";
      statusMessageEl.hidden = true;
    }
  }

  function formatPercent(value) {
    const pct = (value || 0) * 100;
    return `${pct.toFixed(1)}%`;
  }

  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
  }

  // Sequential single-hue tint (blue), lighter -> darker with magnitude,
  // matching the reference palette's sequential ramp. Alpha is capped so
  // text stays readable against the surface in both color schemes.
  function probabilityBackground(value) {
    const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const rgb = isDark ? "57, 135, 229" : "37, 106, 191";
    const alpha = Math.min(Math.max(value, 0), 1) * 0.45;
    if (alpha <= 0.01) return "transparent";
    return `rgba(${rgb}, ${alpha.toFixed(3)})`;
  }

  function findCompetition(slug) {
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

  function buildDateSelect(dates) {
    dateSelectEl.innerHTML = "";
    dates.forEach((date) => {
      const option = document.createElement("option");
      option.value = date;
      option.textContent = formatDateLabel(date);
      dateSelectEl.appendChild(option);
    });
  }

  // Flattens a columns tree into the leaf columns actually rendered in the body
  // (a group's children, in order, or the column itself when it has none) --
  // same left-to-right order the header row(s) below use.
  function leafColumns(columns) {
    return columns.flatMap((column) => column.children || [column]);
  }

  // Default sort when the user hasn't clicked any header yet: classificação
  // order -- team.standings.rank is already the official position (full CBF
  // tiebreak, computed server-side by src.simulation.standings.rank_table),
  // so no client-side tiebreak approximation is needed here.
  function defaultOrderedTeams(teams) {
    return [...teams].sort((a, b) => a.standings.rank - b.standings.rank);
  }

  function getValue(team, column) {
    if (column.kind === "team") return team.team;
    if (column.kind === "int") return team.standings[column.key];
    return team.probs[column.key];
  }

  function compareByColumn(column, teamA, teamB) {
    const a = getValue(teamA, column);
    const b = getValue(teamB, column);
    if (column.kind === "team") return String(a).localeCompare(String(b), "pt-BR");
    return a - b;
  }

  // Sorts by whichever column the user last clicked (persists across
  // date/season/competition switches); falls back to the classificação
  // default if nothing's been clicked yet, or the clicked column doesn't
  // exist in the currently displayed competition (e.g. "Playoff" doesn't
  // exist for Serie A).
  function sortedTeams(teams, columns) {
    if (!state.sort) return defaultOrderedTeams(teams);
    const column = [TEAM_COLUMN, ...leafColumns(columns)].find(
      (c) => c.key === state.sort.key && (c.kind || "percent") === state.sort.kind,
    );
    if (!column) return defaultOrderedTeams(teams);
    const sign = state.sort.direction === "asc" ? 1 : -1;
    return [...teams].sort((a, b) => sign * compareByColumn(column, a, b));
  }

  function isColumnActive(column) {
    const kind = column.kind || "percent";
    if (state.sort) return column.key === state.sort.key && kind === state.sort.kind;
    return column.key === "points"; // classificação default's primary key, for the indicator only
  }

  function columnDirection() {
    return state.sort ? state.sort.direction : "desc";
  }

  function sortArrow(direction) {
    return direction === "asc" ? " ▲" : " ▼";
  }

  function makeSortableHeader(column) {
    const th = document.createElement("th");
    const active = isColumnActive(column);
    const direction = columnDirection(column);
    th.textContent = column.label + (active ? sortArrow(direction) : "");
    th.className = "sortable";
    th.tabIndex = 0;
    th.setAttribute(
      "aria-sort",
      active ? (direction === "asc" ? "ascending" : "descending") : "none",
    );
    th.dataset.sortKey = column.key;
    th.dataset.sortKind = column.kind || "percent";
    return th;
  }

  function handleSortHeaderActivated(th) {
    const key = th.dataset.sortKey;
    const kind = th.dataset.sortKind;
    if (state.sort && state.sort.key === key && state.sort.kind === kind) {
      state.sort.direction = state.sort.direction === "desc" ? "asc" : "desc";
    } else {
      state.sort = { key, kind, direction: kind === "team" ? "asc" : "desc" };
    }
    renderSnapshot();
  }

  tableHeadRowEl.parentElement.addEventListener("click", (event) => {
    const th = event.target.closest("th.sortable");
    if (th) handleSortHeaderActivated(th);
  });

  tableHeadRowEl.parentElement.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const th = event.target.closest("th.sortable");
    if (!th) return;
    event.preventDefault();
    handleSortHeaderActivated(th);
  });

  function renderTableHead(columns) {
    const hasGroups = columns.some((column) => column.children);
    tableHeadRowEl.innerHTML = "";
    tableHeadRowEl.parentElement
      .querySelectorAll("tr.sub-header-row")
      .forEach((row) => row.remove());

    const teamHeader = makeSortableHeader(TEAM_COLUMN);
    if (hasGroups) teamHeader.rowSpan = 2;
    tableHeadRowEl.appendChild(teamHeader);

    let subHeaderRow = null;
    if (hasGroups) {
      subHeaderRow = document.createElement("tr");
      subHeaderRow.className = "sub-header-row";
    }

    columns.forEach((column) => {
      if (column.children) {
        const th = document.createElement("th");
        th.textContent = column.label;
        th.colSpan = column.children.length;
        th.className = "group-header";
        tableHeadRowEl.appendChild(th);
        column.children.forEach((child) => {
          const subTh = makeSortableHeader(child);
          subTh.classList.add("sub-header");
          subHeaderRow.appendChild(subTh);
        });
      } else {
        const th = makeSortableHeader(column);
        if (hasGroups) th.rowSpan = 2;
        tableHeadRowEl.appendChild(th);
      }
    });

    if (subHeaderRow) tableHeadRowEl.parentElement.appendChild(subHeaderRow);
  }

  function renderTable(columns, teams) {
    const leaves = leafColumns(columns);
    const orderedTeams = sortedTeams(teams, columns);
    renderTableHead(columns);
    tableBodyEl.innerHTML = "";

    orderedTeams.forEach((team) => {
      const row = document.createElement("tr");
      // Zone comes straight from the export (src.site.export_site_data's
      // _real_classification) -- already accounts for guaranteed-slot cascades,
      // so it's rendered as-is rather than re-derived from rank here.
      if (team.standings.zone) {
        row.classList.add(`zone-${team.standings.zone.replace(/_/g, "-")}`);
      }

      const teamCell = document.createElement("td");
      teamCell.className = "team-cell";

      const rank = document.createElement("span");
      rank.className = "rank-number";
      rank.textContent = String(team.standings.rank);

      const crest = document.createElement("img");
      crest.className = "crest";
      crest.src = team.crest;
      crest.alt = `Escudo do ${displayTeamName(team.team)}`;
      crest.loading = "lazy";

      const name = document.createElement("span");
      name.className = "team-name";
      name.textContent = displayTeamName(team.team);

      teamCell.appendChild(rank);
      teamCell.appendChild(crest);
      teamCell.appendChild(name);
      row.appendChild(teamCell);

      leaves.forEach((column) => {
        const cell = document.createElement("td");
        if (column.kind === "int") {
          cell.className = "stat-cell";
          cell.textContent = String(team.standings[column.key]);
        } else {
          const value = team.probs[column.key];
          cell.className = "prob-cell";
          cell.textContent = formatPercent(value);
          cell.style.backgroundColor = probabilityBackground(value);
        }
        row.appendChild(cell);
      });

      tableBodyEl.appendChild(row);
    });
  }

  // {zone key -> display label}, prefixed with the group's own label for a
  // spot nested under one (e.g. "Libertadores - Fase de grupos") -- on the
  // table itself that context comes from the group header, but a legend
  // entry has no such header above it. Skips a group's own aggregate-total
  // child (key === the group's key, e.g. "libertadores" under "Libertadores"
  // -- labeled "Geral" on the table): it's a combined probability column, not
  // a real position-based zone, so it never appears as a team's own `zone`.
  function zoneLabels(columns) {
    const labels = {};
    columns.forEach((column) => {
      if (column.children) {
        column.children.forEach((child) => {
          if (child.key !== column.key) labels[child.key] = `${column.label} - ${child.label}`;
        });
      } else {
        labels[column.key] = column.label;
      }
    });
    return labels;
  }

  function renderZoneLegend(columns) {
    const labels = zoneLabels(columns);
    zoneLegendEl.innerHTML = "";
    ZONE_LEGEND_ORDER.forEach((zoneKey) => {
      const label = labels[zoneKey];
      if (!label) return;

      const item = document.createElement("li");
      item.className = "zone-legend-item";

      const swatch = document.createElement("span");
      swatch.className = `zone-legend-swatch zone-${zoneKey.replace(/_/g, "-")}`;

      const text = document.createElement("span");
      text.textContent = label;

      item.appendChild(swatch);
      item.appendChild(text);
      zoneLegendEl.appendChild(item);
    });
  }

  function renderSnapshot() {
    const snapshot = state.seasonData.snapshots[state.date];
    const columns = [STANDINGS_GROUP, ...state.seasonData.columns];
    renderTable(columns, snapshot.teams);
    renderZoneLegend(columns);
  }

  async function loadSeason(slug, season) {
    showStatus("");
    tableBodyEl.innerHTML = '<tr><td class="status-cell">Carregando...</td></tr>';
    try {
      const data = await fetchJSON(`data/${slug}/${season}.json`);
      state.seasonData = data;

      // loadSeason only ever runs from a division/season change (see
      // selectCompetition/selectSeason below) -- always reset to the most
      // recent date rather than trying to carry over whatever was selected
      // before.
      const date = data.dates[data.dates.length - 1];
      buildDateSelect(data.dates);
      dateSelectEl.value = date;
      state.date = date;

      renderSnapshot();
    } catch (error) {
      tableHeadRowEl.innerHTML = "";
      tableBodyEl.innerHTML = "";
      state.seasonData = null;
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
    }
  }

  function selectCompetition(slug) {
    const competition = findCompetition(slug);
    if (!competition) return;

    state.competitionSlug = slug;
    updateTabSelection();
    buildSeasonSelect(competition.seasons);

    // Keep the currently selected season when switching competitions (e.g. stay
    // on 2025 when going from Serie A to Serie B) as long as the new competition
    // actually has that season; only fall back to the latest one otherwise.
    const season = competition.seasons.includes(state.season)
      ? state.season
      : competition.seasons[competition.seasons.length - 1];
    seasonSelectEl.value = String(season);
    state.season = season;

    loadSeason(state.competitionSlug, state.season);
  }

  function selectSeason(season) {
    state.season = season;
    loadSeason(state.competitionSlug, state.season);
  }

  seasonSelectEl.addEventListener("change", (event) => {
    selectSeason(Number(event.target.value));
  });

  dateSelectEl.addEventListener("change", (event) => {
    state.date = event.target.value;
    renderSnapshot();
  });

  async function init() {
    try {
      state.manifest = await fetchJSON("data/manifest.json");
    } catch (error) {
      showStatus(`Não foi possível carregar o manifesto: ${error.message}`);
      return;
    }

    if (!state.manifest.competitions || state.manifest.competitions.length === 0) {
      showStatus("Nenhuma competição disponível.");
      return;
    }

    buildTabs(state.manifest.competitions);
    selectCompetition(state.manifest.competitions[0].slug);
  }

  init();
})();
