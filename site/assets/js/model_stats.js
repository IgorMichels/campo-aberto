(() => {
  "use strict";

  // Rendering only -- no aggregation here. Every number this page shows was
  // already computed by src.site.model_stats.export_model_stats at pipeline-
  // run time into site/data/model_stats.json (see plans/model_stats_page.md's
  // architecture-decision note for why: this reuses the exact same closed-form
  // Dixon-Coles math every played card already shows, ported to Python once
  // instead of re-aggregated in the browser on every page load).

  const SITE_ROOT = "../";
  function siteURL(path) {
    return `${SITE_ROOT}${path}`;
  }

  async function fetchJSON(path) {
    const response = await fetch(siteURL(path));
    if (!response.ok) {
      throw new Error(`Falha ao carregar ${path}: HTTP ${response.status}`);
    }
    return response.json();
  }

  function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function formatBrier(value) {
    return value.toFixed(4);
  }

  // {n, exact_pct, direction_pct, brier, baseline_uniform_brier,
  // baseline_climatology_brier, baseline_favorite_direction_pct} -> the 4
  // headline tiles, computed from the "all" breakdown group.
  function renderKPITiles(containerId, allMetrics) {
    const container = document.getElementById(containerId);
    const tiles = [
      { label: "Partidas avaliadas", value: allMetrics.n.toLocaleString("pt-BR") },
      { label: "Placar exato", value: formatPercent(allMetrics.exact_pct) },
      { label: "Direção (1X2)", value: formatPercent(allMetrics.direction_pct) },
      {
        label: "Brier (vs. baseline uniforme)",
        value: `${formatBrier(allMetrics.brier)} vs. ${formatBrier(
          allMetrics.baseline_uniform_brier,
        )}`,
      },
    ];
    container.innerHTML = tiles
      .map(
        (tile) => `
      <div class="kpi-tile">
        <div class="kpi-value">${tile.value}</div>
        <div class="kpi-label">${tile.label}</div>
      </div>
    `,
      )
      .join("");
  }

  // "Serie A"/"Serie B" is the raw competition field's real value (matches
  // configs/*.yaml's own `name:`, e.g. configs/serie_a_2026.yaml) -- every
  // other page displaying it adds the accent for rendering only, same
  // convention as matches_shared.js's competition-tab labels.
  function displayCompetitionLabel(label) {
    return label.replace("Serie", "Série");
  }

  // One row per breakdown group: "Geral" (all), then each competition, then
  // each competition+season -- same shape src.site.model_stats.aggregate_metrics
  // groups records into. Shared by the breakdown table and the accuracy bar
  // chart, so both list the exact same rows in the exact same order.
  function breakdownRows(breakdown) {
    return [
      { label: "Geral", metrics: breakdown.all },
      ...Object.entries(breakdown.by_competition).map(([label, metrics]) => ({
        label: displayCompetitionLabel(label),
        metrics,
      })),
      ...Object.entries(breakdown.by_competition_season).map(([label, metrics]) => ({
        label: displayCompetitionLabel(label),
        metrics,
      })),
    ];
  }

  function renderBreakdownTable(containerId, breakdown) {
    const rows = breakdownRows(breakdown);

    const bodyRows = rows
      .map(
        ({ label, metrics }) => `
      <tr>
        <td class="team-cell">${label}</td>
        <td>${metrics.n.toLocaleString("pt-BR")}</td>
        <td>${formatPercent(metrics.exact_pct)}</td>
        <td>${formatPercent(metrics.direction_pct)}</td>
        <td>${formatBrier(metrics.brier)}</td>
      </tr>
    `,
      )
      .join("");

    document.getElementById(containerId).innerHTML = `
      <table>
        <thead>
          <tr id="table-head-row">
            <th>Recorte</th>
            <th>N</th>
            <th>Placar exato</th>
            <th>Direção</th>
            <th>Brier</th>
          </tr>
        </thead>
        <tbody>${bodyRows}</tbody>
      </table>
    `;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // model_stats.json's "calibration" array: [{bin_lo, bin_hi, n,
  // mean_predicted, observed_freq}, ...], one entry per equal-width
  // probability bin (see src.models.backtest.calibration_table) -- a bin
  // with n===0 has mean_predicted/observed_freq: null.
  function renderCalibrationChart(containerId, bins) {
    const nonEmpty = bins.filter((bin) => bin.n > 0);

    const gridColor = cssVar("--gridline") || "#EAECF0";
    const borderColor = cssVar("--border") || "#D9DEE7";
    const textPrimary = cssVar("--text-primary") || "#1B2A4A";
    const textSecondary = cssVar("--text-secondary") || "#4A5568";
    const surface = cssVar("--surface-1") || "#FCFCFB";
    const accent = cssVar("--accent") || "#2A78D6";

    // Marker area scales with sqrt(n) (so it's proportional to bin POPULATION,
    // not radius-linear, which would visually overstate small differences) --
    // min-max normalized over just this chart's own bins into a fixed
    // 8-28px diameter range, so the actual n values (a handful to several
    // thousand, see plans/model_stats_page.md) never produce an invisible or
    // page-dominating marker.
    const sqrtNs = nonEmpty.map((bin) => Math.sqrt(bin.n));
    const sqrtMin = Math.min(...sqrtNs);
    const sqrtMax = Math.max(...sqrtNs);
    const markerSizes = sqrtNs.map((s) =>
      sqrtMax === sqrtMin ? 18 : 8 + ((s - sqrtMin) / (sqrtMax - sqrtMin)) * 20,
    );

    const referenceLine = {
      x: [0, 1],
      y: [0, 1],
      mode: "lines",
      line: { color: borderColor, width: 1.5, dash: "dash" },
      hoverinfo: "skip",
      showlegend: false,
    };

    const calibrationTrace = {
      x: nonEmpty.map((bin) => bin.mean_predicted),
      y: nonEmpty.map((bin) => bin.observed_freq),
      mode: "lines+markers",
      line: { color: accent, width: 1.5 },
      marker: { size: markerSizes, color: accent, line: { color: surface, width: 1 } },
      showlegend: false,
      text: nonEmpty.map(
        (bin) =>
          `[${(bin.bin_lo * 100).toFixed(0)}%, ${(bin.bin_hi * 100).toFixed(0)}%)<br>` +
          `n=${bin.n.toLocaleString("pt-BR")}<br>` +
          `Previsto: ${(bin.mean_predicted * 100).toFixed(1)}%<br>` +
          `Observado: ${(bin.observed_freq * 100).toFixed(1)}%`,
      ),
      hovertemplate: "%{text}<extra></extra>",
    };

    const layout = {
      plot_bgcolor: surface,
      paper_bgcolor: surface,
      font: { family: "system-ui, sans-serif", color: textSecondary },
      margin: { l: 60, r: 20, t: 20, b: 55 },
      showlegend: false,
      xaxis: {
        title: { text: "Probabilidade prevista", font: { color: textPrimary } },
        range: [0, 1],
        tickformat: ".0%",
        showgrid: true,
        gridcolor: gridColor,
        zeroline: false,
        showline: true,
        linecolor: borderColor,
      },
      yaxis: {
        title: { text: "Frequência observada", font: { color: textPrimary } },
        range: [0, 1],
        tickformat: ".0%",
        showgrid: true,
        gridcolor: gridColor,
        zeroline: false,
        showline: true,
        linecolor: borderColor,
      },
    };

    Plotly.newPlot(containerId, [referenceLine, calibrationTrace], layout, {
      responsive: true,
      displaylogo: false,
    });
  }

  // Small text panel, straight from allMetrics' baseline_* fields -- no
  // computation here, everything was already computed by
  // src.site.model_stats.aggregate_metrics.
  function renderBaselinesPanel(containerId, allMetrics) {
    document.getElementById(containerId).innerHTML = `
      <div class="baseline-row">
        <span class="baseline-label">Brier</span>
        <span>modelo <strong>${formatBrier(allMetrics.brier)}</strong>
          vs. uniforme ${formatBrier(allMetrics.baseline_uniform_brier)}
          vs. climatologia ${formatBrier(allMetrics.baseline_climatology_brier)}</span>
      </div>
      <div class="baseline-row">
        <span class="baseline-label">Direção (1X2)</span>
        <span>modelo <strong>${formatPercent(allMetrics.direction_pct)}</strong>
          vs. "sempre mandante" ${formatPercent(allMetrics.baseline_favorite_direction_pct)}</span>
      </div>
    `;
  }

  window.ModelStats = {
    fetchJSON,
    renderKPITiles,
    renderBreakdownTable,
    renderCalibrationChart,
    renderBaselinesPanel,
  };
})();
