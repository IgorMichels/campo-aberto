(() => {
  "use strict";

  // Ports scratch/plot_title_evolution.py's chart (see that file for the design
  // rationale: direct endpoint labels instead of a legend, real club colors with
  // a perceptual clash fallback to line dashes) into a page where division,
  // season, spot and up to MAX_TEAMS teams are picked interactively instead of
  // being hardcoded constants.

  const MAX_TEAMS = 6;
  const NUDGE_SPACING = 9; // percentage-point stacking gap between endpoint labels
  const CREST_WIDTH_PX = 26; // rendered crest width, in the fixed-size figure below
  const FIG_W = 1050;
  const FIG_H = 560;
  const MARGIN = { l: 60, r: 210, t: 60, b: 90 };
  // Right-side padding (as a fraction of the season's day span) added past the
  // last data point, roomy enough that the endpoint annotation's x position
  // -- crest offset + label gap -- never lands past the axis range, which would
  // silently clip the label text (Plotly annotations are clipped to xref="x"
  // range, unlike the crest images which sit closer to the last real point).
  const RIGHT_PAD_FRACTION = 0.16;
  const DASH_CYCLE = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];
  const CLASH_THRESHOLD = 15.0; // CIE76 Delta-E below this reads as "same color" in a line chart

  const state = {
    manifest: null,
    competitionSlug: null,
    season: null,
    seasonData: null, // {columns, dates, snapshots}
    spot: null,
    masterTeams: [], // classificação-ordered [{team, crest, color}], for the current competition+season
    selectedTeams: [], // team names, subset of masterTeams, length <= MAX_TEAMS
    showTable: false,
  };

  const crestAspectCache = new Map(); // crest path -> width/height, so re-renders don't reload images

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const spotSelectEl = document.getElementById("spot-select");
  const teamPickerToggleEl = document.getElementById("team-picker-toggle");
  const teamPickerCountEl = document.getElementById("team-picker-count");
  const teamPickerPanelEl = document.getElementById("team-picker-panel");
  const tableToggleEl = document.getElementById("table-toggle");
  const chartEl = document.getElementById("evolution-chart");
  const tableWrapEl = document.getElementById("evolution-table-wrap");
  const tableHeadEl = document.getElementById("evolution-table-head");
  const tableBodyEl = document.getElementById("evolution-table-body");
  const statusMessageEl = document.getElementById("status-message");

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
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // -- CIE76 perceptual color distance (sRGB hex -> CIELAB), for clash detection --

  function hexToRgb(hex) {
    const clean = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(clean.slice(i, i + 2), 16) / 255);
  }

  function srgbChannelToLinear(c) {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }

  function hexToLab(hex) {
    const [r, g, b] = hexToRgb(hex).map(srgbChannelToLinear);
    const x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    const y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    const z = r * 0.0193 + g * 0.1192 + b * 0.9505;
    const [xn, yn, zn] = [0.95047, 1.0, 1.08883];
    const f = (t) => (t > (6 / 29) ** 3 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29);
    const [fx, fy, fz] = [f(x / xn), f(y / yn), f(z / zn)];
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }

  function deltaE76(labA, labB) {
    return Math.sqrt(labA.reduce((sum, v, i) => sum + (v - labB[i]) ** 2, 0));
  }

  // Picks a Plotly dash style per team so two teams whose real colors are
  // perceptually indistinguishable (e.g. two black-and-white clubs) still read
  // as separate lines -- first team with a given color keeps "solid"; each
  // later clashing team gets the next dash style unused among its clashes.
  function assignLineDashes(teamsInOrder, colorByTeam) {
    const labByTeam = new Map(teamsInOrder.map((t) => [t, hexToLab(colorByTeam[t])]));
    const dashByTeam = new Map();
    for (const team of teamsInOrder) {
      const clashingDashes = new Set();
      for (const [other, dash] of dashByTeam) {
        if (deltaE76(labByTeam.get(team), labByTeam.get(other)) < CLASH_THRESHOLD) {
          clashingDashes.add(dash);
        }
      }
      dashByTeam.set(
        team,
        DASH_CYCLE.find((d) => !clashingDashes.has(d)),
      );
    }
    return dashByTeam;
  }

  // Orders teams from best to worst final chance at `spot`. Ties at 0% (once
  // the season is decided) are broken by who was *last* mathematically
  // eliminated -- later last-nonzero date ranks higher, a further tie falls
  // back to that date's probability value.
  function rankByFinalChance(teams, dates) {
    const seriesFor = (team) =>
      dates.map(
        (d) =>
          state.seasonData.snapshots[d].teams.find((t) => t.team === team).probs[state.spot] || 0,
      );
    const sortKey = (team) => {
      const series = seriesFor(team);
      const finalValue = series[series.length - 1];
      let lastNonzeroIndex = -1;
      for (let i = series.length - 1; i >= 0; i--) {
        if (series[i] > 0) {
          lastNonzeroIndex = i;
          break;
        }
      }
      const lastNonzeroDate = lastNonzeroIndex >= 0 ? dates[lastNonzeroIndex] : dates[0];
      const valueAtLastNonzero = lastNonzeroIndex >= 0 ? series[lastNonzeroIndex] : 0;
      return [finalValue, lastNonzeroDate, valueAtLastNonzero];
    };
    return [...teams].sort((a, b) => {
      const [fa, da, va] = sortKey(a);
      const [fb, db, vb] = sortKey(b);
      if (fa !== fb) return fb - fa;
      if (da !== db) return da > db ? -1 : 1;
      return vb - va;
    });
  }

  // Endpoint labels (crest + name + %) are placed at each team's true final
  // value, then nudged just enough to keep a minimum vertical gap between
  // neighbors -- e.g. a title race with a clear favorite and several
  // long-shots clustered near 0% would otherwise render as illegibly
  // overlapping text. `orderedBestToWorst` sets the tie-break order among
  // equal/nearby values.
  //
  // The leader (index 0) is always anchored at its own true value and never
  // moved -- otherwise a low-probability cluster needing room near the floor
  // could shift the *entire* chain upward, pushing the leader's crest above
  // the chart (this happened with e.g. a decided title race: champion at
  // ~100%, every other contender tied at 0%). Contenders are first pushed
  // down just enough to clear the label above them, then pulled back up to
  // stay on-chart, and finally clamped so none of them ever passes the
  // leader.
  function declutterEndpointLabels(orderedBestToWorst, trueYByTeam, minGap, floor = 2, ceil = 103) {
    const y = orderedBestToWorst.map((name) => trueYByTeam.get(name));

    for (let i = 1; i < y.length; i++) {
      if (y[i - 1] - y[i] < minGap) y[i] = y[i - 1] - minGap;
    }
    for (let i = y.length - 1; i >= 1; i--) {
      if (y[i] < floor) y[i] = floor;
      if (i > 1 && y[i - 1] - y[i] < minGap) y[i - 1] = y[i] + minGap;
    }
    for (let i = 1; i < y.length; i++) {
      if (y[0] - y[i] < minGap) y[i] = y[0] - minGap;
    }
    for (let i = 0; i < y.length; i++) y[i] = Math.min(ceil, Math.max(floor, y[i]));

    return new Map(orderedBestToWorst.map((name, i) => [name, y[i]]));
  }

  function leafColumns(columns) {
    return columns.flatMap((column) => column.children || [column]);
  }

  // Spot picker options: a nested column (e.g. "Fase de grupos" under
  // "Libertadores") is labeled with its group for context, since "Geral" alone
  // is ambiguous between Libertadores and Acesso.
  function spotOptions(columns) {
    return columns.flatMap((column) =>
      column.children
        ? column.children.map((child) => ({
            key: child.key,
            label: `${column.label} — ${child.label}`,
          }))
        : [{ key: column.key, label: column.label }],
    );
  }

  // Order used for the team picker dropdown, the chart's dash-assignment
  // order and the table view's columns -- alphabetical, so the picker reads
  // like a normal team list regardless of the current standings.
  function alphabeticalOrderedTeams(teams) {
    return [...teams].sort((a, b) => a.team.localeCompare(b.team, "pt-BR"));
  }

  function loadCrestAspect(src) {
    if (crestAspectCache.has(src)) return Promise.resolve(crestAspectCache.get(src));
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const aspect = img.naturalWidth / img.naturalHeight;
        crestAspectCache.set(src, aspect);
        resolve(aspect);
      };
      img.onerror = () => {
        crestAspectCache.set(src, 1);
        resolve(1);
      };
      img.src = src;
    });
  }

  // ---- competition / season / spot selectors ----

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

  function buildSpotSelect(columns) {
    const options = spotOptions(columns);
    const previousSpot = state.spot;
    spotSelectEl.innerHTML = "";
    options.forEach((option) => {
      const el = document.createElement("option");
      el.value = option.key;
      el.textContent = option.label;
      spotSelectEl.appendChild(el);
    });
    const stillValid = options.some((o) => o.key === previousSpot);
    state.spot = stillValid
      ? previousSpot
      : options.find((o) => o.key === "title")?.key || options[0].key;
    spotSelectEl.value = state.spot;
  }

  // ---- team picker ----

  function teamPickerCountLabel() {
    return `(${state.selectedTeams.length}/${MAX_TEAMS})`;
  }

  function updateTeamPickerDisabledState() {
    teamPickerCountEl.textContent = teamPickerCountLabel();
    const atLimit = state.selectedTeams.length >= MAX_TEAMS;
    teamPickerPanelEl.querySelectorAll(".team-picker-option").forEach((row) => {
      const checkbox = row.querySelector("input");
      const isChecked = state.selectedTeams.includes(checkbox.dataset.team);
      checkbox.disabled = atLimit && !isChecked;
      row.classList.toggle("disabled", checkbox.disabled);
    });
  }

  function buildTeamPicker(masterTeams) {
    teamPickerPanelEl.innerHTML = "";
    masterTeams.forEach(({ team, crest }) => {
      const row = document.createElement("label");
      row.className = "team-picker-option";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.team = team;
      checkbox.checked = state.selectedTeams.includes(team);
      checkbox.addEventListener("change", () => handleTeamToggled(team, checkbox));

      const img = document.createElement("img");
      img.src = crest;
      img.alt = "";
      img.loading = "lazy";

      const name = document.createElement("span");
      name.textContent = team;

      row.appendChild(checkbox);
      row.appendChild(img);
      row.appendChild(name);
      teamPickerPanelEl.appendChild(row);
    });
    updateTeamPickerDisabledState();
  }

  function handleTeamToggled(team, checkbox) {
    if (checkbox.checked) {
      if (state.selectedTeams.length >= MAX_TEAMS) {
        checkbox.checked = false;
        return;
      }
      state.selectedTeams.push(team);
    } else {
      state.selectedTeams = state.selectedTeams.filter((t) => t !== team);
    }
    updateTeamPickerDisabledState();
    renderActiveView();
  }

  function openTeamPicker() {
    teamPickerPanelEl.hidden = false;
    teamPickerToggleEl.setAttribute("aria-expanded", "true");
  }

  function closeTeamPicker() {
    teamPickerPanelEl.hidden = true;
    teamPickerToggleEl.setAttribute("aria-expanded", "false");
  }

  teamPickerToggleEl.addEventListener("click", () => {
    if (teamPickerPanelEl.hidden) openTeamPicker();
    else closeTeamPicker();
  });

  document.addEventListener("click", (event) => {
    if (!document.getElementById("team-picker").contains(event.target)) closeTeamPicker();
  });

  // ---- chart ----

  function monthTicks(dates) {
    const first = dates[0];
    const last = dates[dates.length - 1];
    const ticks = [];
    const cursor = new Date(first.getFullYear(), first.getMonth(), 1);
    while (cursor <= last) {
      ticks.push(new Date(cursor));
      cursor.setMonth(cursor.getMonth() + 1);
    }
    return ticks;
  }

  async function renderChart() {
    chartEl.hidden = false;
    tableWrapEl.hidden = true;
    tableToggleEl.setAttribute("aria-pressed", "false");

    if (state.selectedTeams.length === 0) {
      Plotly.purge(chartEl);
      showStatus("Selecione ao menos um time para visualizar o gráfico.");
      return;
    }
    showStatus("");

    const isoDates = state.seasonData.dates;
    const dates = isoDates.map((d) => new Date(`${d}T00:00:00`));
    const x = dates.map((d) => (d - dates[0]) / 86400000);
    const totalDays = x[x.length - 1] || 1;

    const teamByName = (name) => state.masterTeams.find((t) => t.team === name);
    const colorByTeam = Object.fromEntries(state.masterTeams.map((t) => [t.team, t.color]));

    const rankedTeams = rankByFinalChance(state.selectedTeams, isoDates);
    const lastDate = isoDates[isoDates.length - 1];
    const trueYByTeam = new Map(
      rankedTeams.map((name) => [
        name,
        (state.seasonData.snapshots[lastDate].teams.find((t) => t.team === name).probs[
          state.spot
        ] || 0) * 100,
      ]),
    );

    // Dash assignment order follows the classificação order, not click order,
    // so it stays stable regardless of the sequence teams were selected in.
    const orderedSelected = state.masterTeams
      .map((t) => t.team)
      .filter((t) => state.selectedTeams.includes(t));
    const lineDash = assignLineDashes(orderedSelected, colorByTeam);

    const xaxisRange = [-totalDays * 0.02, totalDays + totalDays * RIGHT_PAD_FRACTION];
    const pxPerXUnit = (FIG_W - MARGIN.l - MARGIN.r) / (xaxisRange[1] - xaxisRange[0]);
    const pxPerYUnit = (FIG_H - MARGIN.t - MARGIN.b) / 105;

    const crestAspects = await Promise.all(
      orderedSelected.map((name) => loadCrestAspect(teamByName(name).crest)),
    );

    // A crest sits vertically centered on its (possibly nudged) label position;
    // keep the label at least half the crest's rendered height away from the
    // 0%/100% edges, or its icon gets cropped by the plot area -- the tallest
    // crest among the selected teams sets how much clearance every label needs.
    const halfCrestHeights = crestAspects.map((aspect) => CREST_WIDTH_PX / aspect / 2 / pxPerYUnit);
    const crestClearance = Math.max(...halfCrestHeights) + 1;
    const endpointDisplayY = declutterEndpointLabels(
      rankedTeams,
      trueYByTeam,
      NUDGE_SPACING,
      crestClearance,
      105 - crestClearance,
    );

    const gridColor = cssVar("--gridline") || "#EAECF0";
    const borderColor = cssVar("--border") || "#D9DEE7";
    const textPrimary = cssVar("--text-primary") || "#1B2A4A";
    const textSecondary = cssVar("--text-secondary") || "#4A5568";
    const surface = cssVar("--surface-1") || "#FCFCFB";

    const traces = orderedSelected.map((name) => {
      const team = teamByName(name);
      const y = isoDates.map(
        (d) =>
          (state.seasonData.snapshots[d].teams.find((t) => t.team === name).probs[state.spot] ||
            0) * 100,
      );
      const markerSizes = new Array(y.length).fill(5);
      markerSizes[markerSizes.length - 1] = 0; // endpoint is replaced by the crest icon
      return {
        x,
        y,
        mode: "lines+markers",
        name,
        line: { color: team.color, width: 2, dash: lineDash.get(name) },
        marker: { size: markerSizes, color: team.color, line: { color: surface, width: 1 } },
        text: isoDates.map(formatDateLabel),
        hovertemplate: `<b>${name}</b><br>%{text}: %{y:.1f}%<extra></extra>`,
      };
    });

    const images = [];
    const annotations = [];
    const shapes = [];
    orderedSelected.forEach((name, i) => {
      const team = teamByName(name);
      const yEnd = trueYByTeam.get(name);
      const yDisplay = endpointDisplayY.get(name);
      const nudge = yDisplay - yEnd;
      const crestX = x[x.length - 1] + totalDays * 0.03;

      if (Math.abs(nudge) > 0.01) {
        shapes.push({
          type: "line",
          x0: x[x.length - 1],
          y0: yEnd,
          x1: crestX,
          y1: yDisplay,
          line: { color: team.color, width: 1 },
          layer: "below",
        });
      }

      const aspect = crestAspects[i];
      const crestHPx = CREST_WIDTH_PX / aspect;
      images.push({
        source: team.crest,
        xref: "x",
        yref: "y",
        x: crestX,
        y: yDisplay,
        xanchor: "center",
        yanchor: "middle",
        sizex: CREST_WIDTH_PX / pxPerXUnit,
        sizey: crestHPx / pxPerYUnit,
        sizing: "contain",
        layer: "above",
      });
      annotations.push({
        x: crestX + CREST_WIDTH_PX / pxPerXUnit / 2 + totalDays * 0.004,
        y: yDisplay,
        text: `<b>${name.split(" / ")[0]}</b>  ${yEnd.toFixed(1)}%`,
        showarrow: false,
        xanchor: "left",
        yanchor: "middle",
        font: { size: 11, color: textPrimary },
      });
    });

    const ticks = monthTicks(dates);
    const spotLabel =
      Array.from(spotSelectEl.options).find((o) => o.value === state.spot)?.textContent ||
      state.spot;

    const layout = {
      title: {
        text: `<b>${spotLabel}</b>`,
        font: { size: 22, color: textPrimary, family: "system-ui, sans-serif" },
        x: 0.02,
        xanchor: "left",
        y: 0.97,
        yanchor: "top",
      },
      plot_bgcolor: surface,
      paper_bgcolor: surface,
      font: { family: "system-ui, sans-serif", color: textSecondary },
      width: FIG_W,
      height: FIG_H,
      margin: MARGIN,
      showlegend: false,
      hovermode: "x",
      images,
      annotations,
      shapes,
      xaxis: {
        tickmode: "array",
        tickvals: ticks.map((d) => (d - dates[0]) / 86400000),
        ticktext: ticks.map((d) =>
          d.toLocaleDateString("pt-BR", { month: "short", year: "numeric" }),
        ),
        tickangle: 90,
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: borderColor,
        range: xaxisRange,
        tickfont: { size: 12 },
        showspikes: true,
        spikemode: "across",
        spikedash: "dot",
        spikecolor: borderColor,
        spikethickness: 1,
      },
      yaxis: {
        showgrid: true,
        gridcolor: gridColor,
        gridwidth: 1,
        zeroline: false,
        showline: false,
        ticksuffix: "%",
        tickfont: { size: 13 },
        range: [0, 105],
      },
    };

    Plotly.react(chartEl, traces, layout, { responsive: false, displaylogo: false });
  }

  function renderTable() {
    chartEl.hidden = true;
    tableWrapEl.hidden = false;
    tableToggleEl.setAttribute("aria-pressed", "true");

    if (state.selectedTeams.length === 0) {
      showStatus("Selecione ao menos um time para visualizar a tabela.");
      tableHeadEl.innerHTML = "";
      tableBodyEl.innerHTML = "";
      return;
    }
    showStatus("");

    const orderedSelected = state.masterTeams
      .map((t) => t.team)
      .filter((t) => state.selectedTeams.includes(t));

    tableHeadEl.innerHTML = "";
    const dateHeader = document.createElement("th");
    dateHeader.textContent = "Data";
    tableHeadEl.appendChild(dateHeader);
    orderedSelected.forEach((name) => {
      const th = document.createElement("th");
      const wrapper = document.createElement("div");
      wrapper.className = "table-team-header";

      const crest = document.createElement("img");
      crest.className = "crest";
      crest.src = state.masterTeams.find((t) => t.team === name).crest;
      crest.alt = "";
      crest.loading = "lazy";

      const label = document.createElement("span");
      label.textContent = name;

      wrapper.appendChild(crest);
      wrapper.appendChild(label);
      th.appendChild(wrapper);
      tableHeadEl.appendChild(th);
    });

    tableBodyEl.innerHTML = "";
    state.seasonData.dates.forEach((isoDate) => {
      const row = document.createElement("tr");
      const dateCell = document.createElement("td");
      dateCell.className = "stat-cell";
      dateCell.textContent = formatDateLabel(isoDate);
      row.appendChild(dateCell);

      orderedSelected.forEach((name) => {
        const cell = document.createElement("td");
        cell.className = "prob-cell";
        const team = state.seasonData.snapshots[isoDate].teams.find((t) => t.team === name);
        cell.textContent = formatPercent(team.probs[state.spot]);
        row.appendChild(cell);
      });
      tableBodyEl.appendChild(row);
    });
  }

  function renderActiveView() {
    if (!state.seasonData) return;
    if (state.showTable) renderTable();
    else renderChart();
  }

  tableToggleEl.addEventListener("click", () => {
    state.showTable = !state.showTable;
    renderActiveView();
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!state.showTable) renderActiveView();
  });

  // ---- season / competition loading ----

  async function loadSeason(slug, season) {
    showStatus("Carregando...");
    try {
      const data = await fetchJSON(`data/${slug}/${season}.json`);
      state.seasonData = data;

      buildSpotSelect(data.columns);

      const latestDate = data.dates[data.dates.length - 1];
      state.masterTeams = alphabeticalOrderedTeams(data.snapshots[latestDate].teams);

      const carriedOver = state.selectedTeams.filter((name) =>
        state.masterTeams.some((t) => t.team === name),
      );
      if (carriedOver.length > 0) {
        state.selectedTeams = carriedOver.slice(0, MAX_TEAMS);
      } else {
        // Default pick: the 3 teams currently most likely for the selected spot
        // -- or, once some are already mathematically at 0%, whichever stayed
        // alive the longest (same tie-break as the endpoint labels' ranking).
        const allTeamNames = state.masterTeams.map((t) => t.team);
        state.selectedTeams = rankByFinalChance(allTeamNames, data.dates).slice(0, 3);
      }

      buildTeamPicker(state.masterTeams);
      renderActiveView();
    } catch (error) {
      state.seasonData = null;
      Plotly.purge(chartEl);
      showStatus(`Não foi possível carregar os dados: ${error.message}`);
    }
  }

  function selectCompetition(slug) {
    const competition = findCompetition(slug);
    if (!competition) return;

    state.competitionSlug = slug;
    updateTabSelection();
    buildSeasonSelect(competition.seasons);

    const season = competition.seasons.includes(state.season)
      ? state.season
      : competition.seasons[competition.seasons.length - 1];
    seasonSelectEl.value = String(season);
    state.season = season;

    loadSeason(state.competitionSlug, state.season);
  }

  seasonSelectEl.addEventListener("change", (event) => {
    state.season = Number(event.target.value);
    loadSeason(state.competitionSlug, state.season);
  });

  spotSelectEl.addEventListener("change", (event) => {
    state.spot = event.target.value;
    renderActiveView();
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
