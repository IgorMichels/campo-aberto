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
  // groups records into.
  function renderBreakdownTable(containerId, breakdown) {
    const rows = [
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

  window.ModelStats = { fetchJSON, renderKPITiles, renderBreakdownTable };
})();
