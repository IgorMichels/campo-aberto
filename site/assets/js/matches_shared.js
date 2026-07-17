(() => {
  "use strict";

  // Shared by all 3 "Jogos" pages (matches/upcoming.html, matches/played.html,
  // matches/simulate.html) -- split out of the former single-page matches.js
  // so the sticker-card rendering, Dixon-Coles bridge, and shared modal only
  // have one implementation. Page-specific state/wiring (which manifest to
  // load, how to paginate, free-pick builder vs. real fixtures) lives in each
  // page's own matches_upcoming.js/matches_played.js/matches_simulate.js.
  //
  // renderStickerCard's card layout (crests, most-likely score, heatmap,
  // win/draw/away bar) is inspired by the sticker cards lflaguardia built
  // for the WorldCup2026 project's previsoes page:
  // https://github.com/BrazilianFootball/WorldCup2026

  // Every one of matches/*.html lives one directory below site/ -- prefixing
  // every fetch/image path here, once, instead of at every call site.
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

  // Canonical team names are dict keys like "Flamengo / RJ" (state suffix
  // kept for search-by-state and params lookups) -- displayTeamName strips
  // the " / UF" suffix for anything actually shown to the user. Shared via
  // team_display.js, loaded before this file on every page.
  const { displayTeamName } = window.CampoAberto;

  // Filename-safe slug for the downloaded sticker PNG.
  function slugify(value) {
    return normalizeName(value)
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  function formatPercent(value) {
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  // A handful of real club colors are literally #000000 -- a legitimate real
  // color identity, not a data bug. readableTextColor keeps the real color
  // for text whenever it's legible and substitutes white only when it would
  // be invisible against the sticker's own near-black background. Threshold
  // picked against the real exported color distribution (see the original
  // matches.js history) -- every real #000000 club is isolated without
  // touching any other legitimately-dark-but-visible club color.
  const READABLE_TEXT_LUMINANCE_THRESHOLD = 0.015;

  function hexToRgb(hex) {
    const match = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? "").trim());
    if (!match) return null;
    const value = parseInt(match[1], 16);
    return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
  }

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

  // Bare "YYYY-MM-DD" -> "DD/MM/AAAA" -- used for the model note and for a
  // played card's reference_date.
  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
  }

  // Full ISO datetime -> "DD/MM HH:MM", local time.
  function formatDateTimeLabel(isoDatetime) {
    const date = new Date(isoDatetime);
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${day}/${month} ${hours}:${minutes}`;
  }

  // Parameterized (not hardcoded to one page's own tab bar) -- both
  // matches_upcoming.js and matches_played.js need their own instance.
  function buildTabs(tabsEl, competitions, onSelectSlug) {
    tabsEl.innerHTML = "";
    competitions.forEach((competition) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tab-button";
      button.textContent = competition.competition.replace("Serie", "Série");
      button.setAttribute("role", "tab");
      button.dataset.slug = competition.slug;
      button.addEventListener("click", () => onSelectSlug(competition.slug));
      tabsEl.appendChild(button);
    });
  }

  function updateTabSelection(tabsEl, selectedSlug) {
    Array.from(tabsEl.children).forEach((button) => {
      const selected = button.dataset.slug === selectedSlug;
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
  }

  function buildSeasonSelect(seasonSelectEl, seasons) {
    seasonSelectEl.innerHTML = "";
    seasons.forEach((season) => {
      const option = document.createElement("option");
      option.value = String(season);
      option.textContent = String(season);
      seasonSelectEl.appendChild(option);
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

  // `actual` (optional {home, away}, both <= 4) marks the real final score's
  // cell with a distinct "actual" class alongside "best" -- used only by
  // matches/played.html's cards. A real score outside the displayed 0-4
  // range is never passed here (see renderStickerCard): highlighting the
  // wrong, clamped cell would be misleading, so those matches only get the
  // literal scoreline text, no cell highlight.
  function renderHeatmap(scores, best, actual) {
    const maxProb = Math.max(...Object.values(scores), 0.01);
    let html = "";

    for (let h = 4; h >= 0; h--) {
      for (let a = 0; a <= 4; a++) {
        const prob = scores[`${h}_${a}`] || 0;
        const isBest = h === best.home && a === best.away;
        const isActual = !!actual && h === actual.home && a === actual.away;
        const alpha = Math.min(1, (prob / maxProb) * 0.7 + 0.1);
        const bg = isBest
          ? "rgba(255, 255, 255, 0.95)"
          : prob === 0
            ? "rgba(255, 255, 255, 0.04)"
            : `rgba(255, 255, 255, ${alpha})`;
        const probText = prob < 0.001 ? "0%" : formatPercent(prob);
        const cellClasses = ["heatmap-cell", isBest ? "best" : "", isActual ? "actual" : ""]
          .filter(Boolean)
          .join(" ");

        html += `
          <div class="${cellClasses}" style="background: ${bg}">
            <div class="prob">${probText}</div>
            <div class="score">${h}x${a}</div>
          </div>
        `;
      }
    }

    return html;
  }

  // What the sticker's top-info bar shows: a played card (has a
  // `final_score`) shows the date it was actually played; a scheduled real
  // fixture shows its date/time; a postponed one shows "Data a definir"; a
  // free-pick card (none of the above) shows nothing.
  function topInfoText(match) {
    if (match.final_score) return match.date ? formatDateTimeLabel(match.date) : "";
    if (match.status === "scheduled") return match.date ? formatDateTimeLabel(match.date) : "";
    if (match.status === "postponed") return "Data a definir";
    return "";
  }

  // Renders one match's probability "sticker": crests, most-likely score,
  // the 25-cell scoreline heatmap and the home/draw/away win-probability
  // bar. `tiers` (from computeStrengthTiers) drives the rarity border.
  // `match.final_score` (optional {home, away}), set only by
  // matches/played.html, additionally shows the real result next to the
  // most-likely score.
  function renderStickerCard(match, tiers) {
    const best = bestScore(match.scores);
    const actual =
      match.final_score && match.final_score.home <= 4 && match.final_score.away <= 4
        ? match.final_score
        : null;
    const heatmapHTML = renderHeatmap(match.scores, best, actual);

    const borderClass = strengthTierClass(match, tiers);

    const searchText = normalizeName(`${match.home_team} ${match.away_team}`);
    const topInfo = topInfoText(match);

    const realScoreHTML = match.final_score
      ? `<div class="real-score">Real: ${match.final_score.home} - ${match.final_score.away}</div>`
      : "";

    return `
      <div class="sticker-wrapper"
        data-search="${escapeHTML(searchText)}"
        data-home-team="${escapeHTML(match.home_team)}"
        data-away-team="${escapeHTML(match.away_team)}">
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
                <div class="sticker-crest"><img src="${siteURL(
                  match.home_crest,
                )}" alt="${escapeHTML(displayTeamName(match.home_team))}"></div>
                <div class="team-name">${escapeHTML(displayTeamName(match.home_team))}</div>
              </div>

              <div class="score-center">
                <div class="most-likely">${best.home} - ${best.away}</div>
                <div class="most-likely-prob">${formatPercent(best.prob)}</div>
                ${realScoreHTML}
              </div>

              <div class="team">
                <div class="sticker-crest"><img src="${siteURL(
                  match.away_crest,
                )}" alt="${escapeHTML(displayTeamName(match.away_team))}"></div>
                <div class="team-name">${escapeHTML(displayTeamName(match.away_team))}</div>
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

  // The single shared computation for every card type (real upcoming
  // fixture, played match, free pick) -- see plans/confrontos_rework.md
  // Step 5 and plans/immutable-pondering-journal.md. `base` carries at least
  // home_team/away_team (canonical "Team / UF" keys) plus whatever
  // pass-through fields the caller wants preserved (crest/color/date/etc.).
  // `params` (NOT read from any shared module state -- explicit argument)
  // is either the page's one current params.json (upcoming/free-pick) or a
  // played match's own embedded historical snapshot (see
  // src.site.export_matches_data._played_cards) -- either way, shaped
  // {model, shared, teams}. Dispatches through window.ScoreModels[params.model]
  // (see score_models.js) instead of calling one hardcoded implementation,
  // so a candidate model needs no change here. Returns null (caller
  // skips/warns) if either team has no known params.
  function computeCard(base, params) {
    const homeParams = params.teams[base.home_team];
    const awayParams = params.teams[base.away_team];
    if (!homeParams || !awayParams) return null;

    const model = window.ScoreModels[params.model];
    const { muHome, muAway } = model.matchRates(homeParams, awayParams, params.shared);
    const { grid, home_win, draw, away_win } = model.scorelineProbabilities(
      muHome,
      muAway,
      params.shared,
    );

    const scores = {};
    grid.forEach((row, home) => {
      row.forEach((prob, away) => {
        scores[`${home}_${away}`] = prob;
      });
    });

    // Symmetric in home/away -- measures "how strong are these two teams
    // combined", independent of who hosts. Drives the card's rarity border
    // (see computeStrengthTiers/strengthTierClass).
    const total_strength = model.teamStrength(homeParams) + model.teamStrength(awayParams);

    return { ...base, scores, home_win, draw, away_win, total_strength };
  }

  // Rarity-tier cutoffs for the card border, derived from the *current*
  // data's own distribution of total_strength rather than hardcoded -- see
  // plans/confrontos_rework.md's "Rarity restyle" section. Computed once per
  // page load (right after that page's own params load) and passed
  // explicitly to renderStickerCard/strengthTierClass from then on (no
  // shared module-level state, since matches/played.html computes a card's
  // total_strength from a DIFFERENT params object per match, but still
  // wants one consistent tier scale across the whole page -- see each
  // page's own init()). `model` selects window.ScoreModels[model].teamStrength
  // (see score_models.js) -- never dereferenced when paramsTeams is empty
  // (a real case: a played season with no match that has a model yet), so a
  // null/missing model is harmless in that case.
  function computeStrengthTiers(paramsTeams, model) {
    const scoreModel = window.ScoreModels[model];
    const teamNames = Object.keys(paramsTeams);
    const totals = [];
    for (let i = 0; i < teamNames.length; i++) {
      const a = paramsTeams[teamNames[i]];
      for (let j = i + 1; j < teamNames.length; j++) {
        const b = paramsTeams[teamNames[j]];
        totals.push(scoreModel.teamStrength(a) + scoreModel.teamStrength(b));
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
          team_strength: scoreModel.teamStrength(paramsTeams[name]),
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
  // computeStrengthTiers (now an explicit parameter, not module state -- see
  // that function's docstring). Checks the categorical "Prime Icon Moments"
  // rule first, then falls through to the ordinary total_strength
  // percentile lookup (Prime/Mid/Base).
  function strengthTierClass(match, tiers) {
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

  // Toggles visibility (not re-rendering) of every currently-rendered card
  // in gridEl based on a search query -- shared by matches/upcoming.html and
  // matches/played.html. Callers pair this with createPaginatedGrid's
  // showAll()/reset() so a non-empty query bypasses pagination for the
  // duration of the search (see each page's own wiring).
  function applyFilterToGrid(gridEl, query) {
    const normalized = normalizeName(query);
    gridEl.querySelectorAll(".sticker-wrapper").forEach((wrapper) => {
      const matches = !normalized || wrapper.dataset.search.includes(normalized);
      wrapper.style.display = matches ? "" : "none";
    });
  }

  // Generic "N at a time, load more" pagination over an already-fetched
  // full item list -- no new network round-trip on "mostrar mais", since
  // export_matches_data.py now ships every remaining/played match up front
  // (see plans/immutable-pondering-journal.md). No precedent for this
  // pattern existed anywhere in the codebase before this split.
  function createPaginatedGrid({ gridEl, moreButtonEl, pageSize, renderItem }) {
    let items = [];
    let visibleCount = 0;

    function render() {
      gridEl.innerHTML = items.slice(0, visibleCount).map(renderItem).join("");
      moreButtonEl.hidden = visibleCount >= items.length;
    }

    function setItems(newItems) {
      items = newItems;
      visibleCount = Math.min(pageSize, items.length);
      render();
    }

    function showMore() {
      visibleCount = Math.min(items.length, visibleCount + pageSize);
      render();
    }

    // Reveals every item at once -- used while a search filter is active,
    // so a match further down the full list than the current page can still
    // be found (see applyFilterToGrid).
    function showAll() {
      visibleCount = items.length;
      render();
    }

    // Reverts to the first page -- used when a search filter is cleared.
    function reset() {
      visibleCount = Math.min(pageSize, items.length);
      render();
    }

    moreButtonEl.addEventListener("click", showMore);

    return { setItems, showAll, reset };
  }

  function updateModelNote(modelNoteEl, referenceDate) {
    if (!referenceDate) {
      modelNoteEl.textContent = "";
      return;
    }
    modelNoteEl.textContent = `Baseado no modelo ajustado até ${formatDateLabel(referenceDate)}.`;
  }

  // ── Shared sticker modal (zoom/download/share/close) ──
  // Markup (#sticker-modal + overlay/toolbar) is duplicated verbatim in all
  // 3 pages (plain inert HTML, no templating available without a build
  // step) -- this is the one shared implementation behind it.

  const modalEl = document.getElementById("sticker-modal");
  const modalOverlayEl = document.getElementById("sticker-modal-overlay");
  const modalContentEl = document.getElementById("sticker-modal-content");
  const modalInnerEl = document.querySelector(".sticker-modal-inner");
  const modalCloseEl = document.getElementById("sticker-modal-close");
  const modalDownloadEl = document.getElementById("sticker-modal-download");
  const modalShareEl = document.getElementById("sticker-modal-share");

  // Default/max zoom applied to the sticker card in the modal (matches the
  // old fixed `transform: scale(1.5)`) -- shrunk by updateModalScale below
  // whenever the viewport is too small to fit it at that zoom.
  const MODAL_MAX_SCALE = 1.5;
  const MODAL_VIEWPORT_FIT_RATIO = 0.92;

  // Sizes .sticker-modal-inner's zoom to fit the current viewport.
  // offsetWidth/scrollHeight reflect the element's *unscaled* layout box
  // (CSS transform doesn't affect layout size), so they're a stable base to
  // scale from regardless of whatever --modal-scale was previously set to.
  function updateModalScale() {
    if (!modalInnerEl || !modalEl.classList.contains("active")) return;
    modalInnerEl.style.setProperty("--modal-scale", "1");
    const naturalWidth = modalInnerEl.offsetWidth;
    const naturalHeight = modalInnerEl.scrollHeight;
    if (!naturalWidth || !naturalHeight) return;
    const fitScale = Math.min(
      (window.innerWidth * MODAL_VIEWPORT_FIT_RATIO) / naturalWidth,
      (window.innerHeight * MODAL_VIEWPORT_FIT_RATIO) / naturalHeight,
    );
    modalInnerEl.style.setProperty("--modal-scale", String(Math.min(MODAL_MAX_SCALE, fitScale)));
  }
  window.addEventListener("resize", updateModalScale);

  // Home/away *display* names of whichever sticker is currently open --
  // used only to build the downloaded PNG's filename.
  let modalTeamNames = null;

  // Flat {key: value, ...} object describing whichever sticker is currently
  // open, merged with home_team/away_team -- passed straight into
  // `new URLSearchParams(...)` by shareStickerLink, so its shape is
  // deliberately generic/caller-defined instead of a hardcoded set of keys:
  // matches/upcoming.html and matches/simulate.html use
  // {home_slug, home_season, away_slug, away_season}, matches/played.html
  // uses {slug, season} (a played match is always single-competition/
  // single-season on both sides) -- see each page's own click wiring.
  let modalShareContext = null;

  function openStickerModal(wrapperEl, context) {
    const container = wrapperEl.querySelector(".sticker-container");
    if (!container) return;
    modalTeamNames = {
      home: displayTeamName(wrapperEl.dataset.homeTeam),
      away: displayTeamName(wrapperEl.dataset.awayTeam),
    };
    modalShareContext = context
      ? { ...context, home_team: wrapperEl.dataset.homeTeam, away_team: wrapperEl.dataset.awayTeam }
      : null;
    modalContentEl.innerHTML = container.outerHTML;
    modalEl.classList.add("active");
    updateModalScale();
  }

  // Same as openStickerModal, but for a card object with no DOM wrapper yet
  // (the shared-link restore path) -- renders it into a detached container
  // with the same renderStickerCard() every other card uses.
  function openStickerModalForCard(card, tiers, context) {
    const detached = document.createElement("div");
    detached.innerHTML = renderStickerCard(card, tiers);
    const wrapper = detached.querySelector(".sticker-wrapper");
    if (wrapper) openStickerModal(wrapper, context);
  }

  function closeStickerModal() {
    modalEl.classList.remove("active");
    modalContentEl.innerHTML = "";
    modalTeamNames = null;
    modalShareContext = null;
  }

  // Collects every same-origin stylesheet rule already loaded on the page,
  // as text -- used to re-apply the site's CSS inside the detached SVG
  // snapshot that renderStickerToPNG rasterizes.
  function collectStylesheetText() {
    let css = "";
    for (const sheet of document.styleSheets) {
      try {
        for (const rule of sheet.cssRules) {
          css += `${rule.cssText}\n`;
        }
      } catch (error) {
        // Cross-origin stylesheet (e.g. a CDN font) -- cssRules is unreadable, skip it.
      }
    }
    return css;
  }

  // Rasterizes a .sticker-container node to a PNG data URL via an SVG
  // <foreignObject> snapshot rather than html2canvas: html2canvas
  // re-implements text layout itself and badly mis-measures letter-spacing
  // and glyph advances (visible as stretched-out text, e.g. "R e m o"), and
  // it drops the background's saturate() filter -- so the downloaded card
  // looked visibly different from the on-screen one. Drawing an <img> of an
  // SVG snapshot instead makes the browser's own renderer lay out the text
  // and apply the filters, matching the on-screen card exactly.
  async function renderStickerToPNG(container, scale = 2) {
    const width = container.offsetWidth;
    const height = container.offsetHeight;

    const clone = container.cloneNode(true);
    for (const img of clone.querySelectorAll("img")) {
      const response = await fetch(img.src);
      const blob = await response.blob();
      img.src = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    const xhtmlNS = "http://www.w3.org/1999/xhtml";
    const wrapper = document.createElementNS(xhtmlNS, "div");
    const style = document.createElementNS(xhtmlNS, "style");
    style.textContent = collectStylesheetText();
    wrapper.appendChild(style);
    wrapper.appendChild(clone);

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("xmlns", svgNS);
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    const foreignObject = document.createElementNS(svgNS, "foreignObject");
    foreignObject.setAttribute("width", "100%");
    foreignObject.setAttribute("height", "100%");
    foreignObject.appendChild(wrapper);
    svg.appendChild(foreignObject);

    const svgDataURL = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
      new XMLSerializer().serializeToString(svg),
    )}`;
    const image = new Image();
    image.src = svgDataURL;
    await image.decode();

    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.drawImage(image, 0, 0, width, height);
    return canvas.toDataURL("image/png");
  }

  async function downloadStickerAsPNG() {
    const container = modalContentEl.querySelector(".sticker-container");
    if (!container) return;
    const dataURL = await renderStickerToPNG(container);
    const home = modalTeamNames ? modalTeamNames.home : "time-casa";
    const away = modalTeamNames ? modalTeamNames.away : "time-visitante";
    const link = document.createElement("a");
    link.download = `${slugify(home)}-x-${slugify(away)}.png`;
    link.href = dataURL;
    link.click();
  }

  let shareFeedbackTimeout = null;
  function flashShareFeedback(message) {
    window.clearTimeout(shareFeedbackTimeout);
    const original = modalShareEl.dataset.originalLabel || modalShareEl.innerHTML;
    modalShareEl.dataset.originalLabel = original;
    modalShareEl.textContent = message;
    shareFeedbackTimeout = window.setTimeout(() => {
      modalShareEl.innerHTML = original;
    }, 1800);
  }

  function shareStickerLink() {
    if (!modalShareContext) return;
    const url = new URL(location.href);
    url.search = new URLSearchParams(modalShareContext).toString();
    const link = url.toString();
    const title = modalTeamNames
      ? `${modalTeamNames.home} x ${modalTeamNames.away} - Campo Aberto`
      : "Jogo - Campo Aberto";

    if (navigator.share) {
      navigator.share({ url: link, title }).catch(() => {});
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(link)
        .then(() => flashShareFeedback("Link copiado!"))
        .catch(() => window.prompt("Copie o link:", link));
      return;
    }
    window.prompt("Copie o link:", link);
  }

  modalOverlayEl.addEventListener("click", closeStickerModal);
  modalCloseEl.addEventListener("click", closeStickerModal);
  modalDownloadEl.addEventListener("click", downloadStickerAsPNG);
  modalShareEl.addEventListener("click", shareStickerLink);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeStickerModal();
  });

  // One team's {team, crest, color} off data/${slug}/${season}.json's latest
  // snapshot -- used by the free-pick builder's roster loading and by the
  // upcoming/free-pick shared-link restore flow.
  async function fetchRosterEntry(slug, season, teamName) {
    const data = await fetchJSON(`data/${slug}/${season}.json`);
    const lastDate = data.dates[data.dates.length - 1];
    return data.snapshots[lastDate].teams.find((team) => team.team === teamName) || null;
  }

  // Counterpart to shareStickerLink for the upcoming/free-pick shared-link
  // shape (?home_slug&home_season&away_slug&away_season&home_team&away_team)
  // -- shared by matches_upcoming.js and matches_simulate.js, both of which
  // reconstruct from CURRENT params.json (unlike matches/played.html, which
  // has its own reconstruction, see matches_played.js). A no-op (besides a
  // console warning) if the params are missing/malformed/stale.
  async function openSharedStickerFromURLViaParams(params, tiers) {
    const searchParams = new URLSearchParams(location.search);
    const homeSlug = searchParams.get("home_slug");
    const homeSeason = searchParams.get("home_season");
    const awaySlug = searchParams.get("away_slug");
    const awaySeason = searchParams.get("away_season");
    const homeTeam = searchParams.get("home_team");
    const awayTeam = searchParams.get("away_team");
    if (!homeSlug || !homeSeason || !awaySlug || !awaySeason || !homeTeam || !awayTeam) return;

    try {
      const [homeEntry, awayEntry] = await Promise.all([
        fetchRosterEntry(homeSlug, Number(homeSeason), homeTeam),
        fetchRosterEntry(awaySlug, Number(awaySeason), awayTeam),
      ]);
      if (!homeEntry || !awayEntry) {
        console.warn("Link compartilhado aponta para um confronto que não existe mais.");
        return;
      }

      const card = computeCard(
        {
          home_team: homeEntry.team,
          away_team: awayEntry.team,
          home_crest: homeEntry.crest,
          away_crest: awayEntry.crest,
          home_color: homeEntry.color,
          away_color: awayEntry.color,
        },
        params,
      );
      if (!card) {
        console.warn("Link compartilhado aponta para um confronto sem parâmetros do modelo.");
        return;
      }

      openStickerModalForCard(card, tiers, {
        home_slug: homeSlug,
        home_season: String(homeSeason),
        away_slug: awaySlug,
        away_season: String(awaySeason),
      });
    } catch (error) {
      console.warn(`Não foi possível abrir a figurinha compartilhada: ${error.message}`);
    }
  }

  window.MatchesShared = {
    siteURL,
    fetchJSON,
    escapeHTML,
    normalizeName,
    slugify,
    formatPercent,
    readableTextColor,
    formatDateLabel,
    formatDateTimeLabel,
    buildTabs,
    updateTabSelection,
    buildSeasonSelect,
    bestScore,
    renderHeatmap,
    renderStickerCard,
    computeCard,
    computeStrengthTiers,
    strengthTierClass,
    applyFilterToGrid,
    createPaginatedGrid,
    updateModelNote,
    openStickerModal,
    openStickerModalForCard,
    closeStickerModal,
    downloadStickerAsPNG,
    shareStickerLink,
    flashShareFeedback,
    fetchRosterEntry,
    openSharedStickerFromURLViaParams,
  };
})();
