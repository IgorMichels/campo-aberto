(() => {
  "use strict";

  const state = {
    manifest: null,
    competitionSlug: null,
    season: null,
    roundData: null, // {round, note, matches} for the currently loaded competition+season
    filterText: "",
  };

  const tabsEl = document.getElementById("competition-tabs");
  const seasonSelectEl = document.getElementById("season-select");
  const roundLabelEl = document.getElementById("round-label");
  const teamFilterEl = document.getElementById("team-filter");
  const stickersGridEl = document.getElementById("stickers-grid");
  const scrollLeftEl = document.getElementById("scroll-left");
  const scrollRightEl = document.getElementById("scroll-right");
  const statusMessageEl = document.getElementById("status-message");
  const modalEl = document.getElementById("sticker-modal");
  const modalOverlayEl = document.getElementById("sticker-modal-overlay");
  const modalContentEl = document.getElementById("sticker-modal-content");

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

  function formatDateLabel(isoDate) {
    const [year, month, day] = isoDate.split("-");
    return `${day}/${month}/${year}`;
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

  // Renders one match's probability "sticker": crests, most-likely score,
  // the 25-cell scoreline heatmap and the home/draw/away win-probability bar.
  function renderStickerCard(match) {
    const best = bestScore(match.scores);
    const heatmapHTML = renderHeatmap(match.scores, best);

    const maxGoals = Math.max(best.home, best.away);
    const borderClass = maxGoals >= 3 ? "border-purple" : maxGoals === 2 ? "border-orange" : "";

    const randomRot = (Math.random() * 8 - 4).toFixed(1);
    const randomY = (Math.random() * 12 - 6).toFixed(1);
    const searchText = normalizeName(`${match.home_team} ${match.away_team}`);

    return `
      <div class="sticker-wrapper"
        style="--rand-rot: ${randomRot}deg; --rand-y: ${randomY}px;"
        data-search="${escapeHTML(searchText)}">
        <div class="sticker-container ${borderClass}" style="--home-color: ${
          match.home_color
        }; --away-color: ${match.away_color};">
          <div class="sticker-bg-blur"></div>

          <div class="sticker-glass">
            <div class="sticker-top-info">
              <span class="info-round">${escapeHTML(state.roundData.round)}</span>
              ${match.date ? `<span> &bull; ${escapeHTML(formatDateLabel(match.date))}</span>` : ""}
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
                <div class="f-bar home" style="width: ${match.home_win * 100}%"></div>
                <div class="f-bar draw" style="width: ${match.draw * 100}%"></div>
                <div class="f-bar away" style="width: ${match.away_win * 100}%"></div>
              </div>
              <div class="footer-stats-row">
                <div class="f-stat home">${formatPercent(match.home_win)}</div>
                <div class="f-stat draw">${formatPercent(match.draw)}</div>
                <div class="f-stat away">${formatPercent(match.away_win)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Renders every card once; the search box then only toggles visibility
  // (applyFilter) instead of re-rendering, so cards keep their random
  // rotation/offset and scroll position while the user types.
  function renderAllStickers() {
    const matches = state.roundData ? state.roundData.matches : [];
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

  async function loadRound(slug, season) {
    showStatus("");
    roundLabelEl.textContent = "";
    stickersGridEl.innerHTML = "";
    try {
      const data = await fetchJSON(`data/${slug}/stickers_${season}.json`);
      state.roundData = data;
      roundLabelEl.textContent = data.round;
      state.filterText = "";
      teamFilterEl.value = "";
      renderAllStickers();
    } catch (error) {
      state.roundData = null;
      showStatus(`Não foi possível carregar os confrontos: ${error.message}`);
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

    loadRound(state.competitionSlug, state.season);
  }

  seasonSelectEl.addEventListener("change", (event) => {
    state.season = Number(event.target.value);
    loadRound(state.competitionSlug, state.season);
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
