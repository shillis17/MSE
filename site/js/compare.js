document.addEventListener("DOMContentLoaded", () => {
  const trackId = searchParams().get("track_id");

  const pageTitle = document.getElementById("pageTitle");
  const pageLead = document.getElementById("pageLead");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const compareLayout = document.getElementById("compareLayout");

  const entityLinks = document.getElementById("entityLinks");
  const songTitle = document.getElementById("songTitle");
  const songArtist = document.getElementById("songArtist");
  const songGenres = document.getElementById("songGenres");
  const spectrogramImage = document.getElementById("spectrogramImage");
  const spectrogramPlayhead = document.getElementById("spectrogramPlayhead");
  const audioPlayer = document.getElementById("audioPlayer");
  const metaGrid = document.getElementById("metaGrid");

  const leftModelSelect = document.getElementById("leftModelSelect");
  const rightModelSelect = document.getElementById("rightModelSelect");
  const leftMeta = document.getElementById("leftMeta");
  const rightMeta = document.getElementById("rightMeta");
  const leftList = document.getElementById("leftList");
  const rightList = document.getElementById("rightList");

  let resizeSyncHandle = null;
  let availableModels = [];
  let leftModel = "panns";
  let rightModel = "clap";
  let currentTrack = null;

  function addMeta(label, value) {
    const item = document.createElement("div");
    item.className = "meta-item";
    item.innerHTML = `<span class="meta-label">${label}</span><div class="meta-value">${value}</div>`;
    metaGrid.appendChild(item);
  }

  function linkOrText(href, label) {
    if (!href) return escapeHtml(label);
    return `<a href="${href}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
  }

  function resetSpectrogramPlayhead() {
    spectrogramPlayhead.style.left = "0%";
    spectrogramPlayhead.classList.add("hidden");
  }

  function updateSpectrogramPlayhead() {
    const duration = audioPlayer.duration;
    const current = audioPlayer.currentTime;

    if (!Number.isFinite(duration) || duration <= 0) {
      resetSpectrogramPlayhead();
      return;
    }

    const progress = Math.max(0, Math.min(1, current / duration));
    spectrogramPlayhead.style.left = `${progress * 100}%`;
    spectrogramPlayhead.classList.remove("hidden");
  }

  function dedupeAndFilterRecommendations(rawList, baseTrack) {
    const seen = new Set();

    return rawList.filter((item) => {
      const artist = String(item.artist || "").toLowerCase().trim();
      const title = String(item.title || "").toLowerCase().trim();
      const baseArtist = String(baseTrack.artist || "").toLowerCase().trim();
      const baseTitle = String(baseTrack.title || "").toLowerCase().trim();

      if (artist === baseArtist && title === baseTitle) {
        return false;
      }

      const key = `${artist}::${title}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function renderCompareCard(track, index, modelName) {
    const t = normalizeTrack(track);
    const rankText = `#${index + 1}`;
    const scoreText = Number.isFinite(Number(t.score))
      ? `${(Number(t.score) * 100).toFixed(2)}% similar`
      : "";

    return `
      <article class="card card-tight">
        <div class="similar-topline">
          <a class="card-title" href="song.html?track_id=${encodeURIComponent(t.track_id)}&model=${encodeURIComponent(modelName)}">
            ${escapeHtml(t.title)}
          </a>
          <span class="similar-rank">
            ${rankText}${scoreText ? ` · ${escapeHtml(scoreText)}` : ""}
          </span>
        </div>
        <div class="card-meta">
          <a href="artist.html?name=${encodeURIComponent(t.artist)}">${escapeHtml(t.artist)}</a>
          ${t.album ? ` · <a href="album.html?name=${encodeURIComponent(t.album)}">${escapeHtml(t.album)}</a>` : ""}
          ${t.duration ? ` · ${escapeHtml(formatDuration(t.duration) || t.duration)}` : ""}
        </div>
        <div class="genre-list">${renderGenres(t.genres)}</div>
      </article>
    `;
  }

  function renderRecommendationColumn(el, metaEl, rawList, baseTrack, modelName) {
    const cleaned = dedupeAndFilterRecommendations(rawList, baseTrack);
    metaEl.textContent = `${cleaned.length} recommendation(s) shown`;

    el.innerHTML = cleaned.length
      ? cleaned.map((item, index) => renderCompareCard(item, index, modelName)).join("")
      : `<div class="compare-empty">No recommendations available.</div>`;
  }

  function syncCompareHeights() {
    if (window.innerWidth <= 980) {
      [...leftList.querySelectorAll(".card"), ...rightList.querySelectorAll(".card")].forEach((card) => {
        card.style.height = "";
      });
      return;
    }

    const leftCards = Array.from(leftList.querySelectorAll(".card"));
    const rightCards = Array.from(rightList.querySelectorAll(".card"));
    const maxLen = Math.max(leftCards.length, rightCards.length);

    [...leftCards, ...rightCards].forEach((card) => {
      card.style.height = "";
    });

    for (let i = 0; i < maxLen; i++) {
      const left = leftCards[i];
      const right = rightCards[i];

      if (!left || !right) continue;

      const height = Math.max(left.offsetHeight, right.offsetHeight);
      left.style.height = `${height}px`;
      right.style.height = `${height}px`;
    }
  }

  function requestSyncCompareHeights() {
    if (resizeSyncHandle) {
      cancelAnimationFrame(resizeSyncHandle);
    }

    resizeSyncHandle = requestAnimationFrame(() => {
      syncCompareHeights();
      resizeSyncHandle = null;
    });
  }

  function renderSharedTrack(track) {
    pageTitle.textContent = `${track.title} · Compare Models`;
    pageLead.textContent = "Compare recommendation output from different similarity models for this track.";

    songTitle.textContent = track.title;
    songArtist.innerHTML = `<a href="artist.html?name=${encodeURIComponent(track.artist)}">${escapeHtml(track.artist)}</a>`;
    songGenres.innerHTML = renderGenres(track.genres);

    entityLinks.innerHTML = `
      <a class="chip-link" href="artist.html?name=${encodeURIComponent(track.artist)}">Artist</a>
      ${track.album ? `<a class="chip-link" href="album.html?name=${encodeURIComponent(track.album)}">Album</a>` : ""}
      ${track.track_url ? `<a class="chip-link" href="${track.track_url}" target="_blank" rel="noreferrer">FMA Page</a>` : ""}
    `;

    if (track.spectrogram_url) {
      spectrogramImage.src = track.spectrogram_url;
      spectrogramImage.alt = `${track.title} spectrogram`;
      spectrogramImage.classList.remove("hidden");
    } else {
      spectrogramImage.classList.add("hidden");
    }

    if (track.audio_url) {
      audioPlayer.src = track.audio_url;
      audioPlayer.classList.remove("hidden");
    } else {
      audioPlayer.classList.add("hidden");
    }

    metaGrid.innerHTML = "";
    addMeta("Track ID", escapeHtml(track.track_id));
    addMeta("Title", escapeHtml(track.title));
    addMeta("Artist", `<a href="artist.html?name=${encodeURIComponent(track.artist)}">${escapeHtml(track.artist)}</a>`);
    addMeta("Album", track.album ? `<a href="album.html?name=${encodeURIComponent(track.album)}">${escapeHtml(track.album)}</a>` : "Unknown album");
    addMeta("Duration", escapeHtml(formatDuration(track.duration) || track.duration || "Unknown"));
    addMeta("License", track.license_title ? linkOrText(track.license_url, track.license_title) : "Unknown");
    addMeta("Artist Website", track.artist_website ? linkOrText(track.artist_website, track.artist_website) : "Not provided");

    resetSpectrogramPlayhead();
  }

  async function loadAvailableModels() {
    try {
      const rootData = await loadRootInfo();
      availableModels = normalizeModelOptions(rootData.available_models);

      const defaultModel = rootData.default_model || availableModels[0]?.name || "panns";

      leftModel = availableModels.find((modelOption) => modelOption.name === defaultModel)?.name || availableModels[0]?.name || "panns";
      rightModel = availableModels.find((modelOption) => modelOption.name !== leftModel)?.name || leftModel;

      leftModelSelect.innerHTML = availableModels.map((modelOption) => `
        <option value="${escapeHtml(modelOption.name)}" ${modelOption.name === leftModel ? "selected" : ""}>
          ${escapeHtml(modelOption.label)}
        </option>
      `).join("");

      rightModelSelect.innerHTML = availableModels.map((modelOption) => `
        <option value="${escapeHtml(modelOption.name)}" ${modelOption.name === rightModel ? "selected" : ""}>
          ${escapeHtml(modelOption.label)}
        </option>
      `).join("");
    } catch (error) {
      console.error(error);
      availableModels = ["panns", "clap"];
      leftModel = "panns";
      rightModel = "clap";

      leftModelSelect.innerHTML = '<option value="panns" selected>PANNS</option><option value="clap">CLAP</option>';
      rightModelSelect.innerHTML = '<option value="panns">PANNS</option><option value="clap" selected>CLAP</option>';
    }
  }

  async function loadModelData(model) {
    const data = await fetchJson(`${SITE_CONFIG.API_BASE}/tracks/${encodeURIComponent(trackId)}?model=${encodeURIComponent(model)}`);
    return data;
  }

  async function refreshComparison() {
    try {
      const [leftData, rightData] = await Promise.all([
        loadModelData(leftModel),
        loadModelData(rightModel)
      ]);

      if (!currentTrack) {
        currentTrack = normalizeTrack(leftData);
        renderSharedTrack(currentTrack);
      }

      const leftRaw = Array.isArray(
        leftData.recommendations || leftData.similar_tracks || leftData.results || leftData.neighbors
      )
        ? (leftData.recommendations || leftData.similar_tracks || leftData.results || leftData.neighbors).map(normalizeTrack)
        : [];

      const rightRaw = Array.isArray(
        rightData.recommendations || rightData.similar_tracks || rightData.results || rightData.neighbors
      )
        ? (rightData.recommendations || rightData.similar_tracks || rightData.results || rightData.neighbors).map(normalizeTrack)
        : [];

      renderRecommendationColumn(leftList, leftMeta, leftRaw, currentTrack, leftModel);
      renderRecommendationColumn(rightList, rightMeta, rightRaw, currentTrack, rightModel);

      requestSyncCompareHeights();
    } catch (error) {
      console.error(error);
      throw error;
    }
  }

  async function loadComparePage() {
    if (!trackId) {
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      errorState.textContent = "Missing track_id parameter.";
      pageTitle.textContent = "Track not specified";
      pageLead.textContent = "Open this page from a song link.";
      return;
    }

    try {
      await loadAvailableModels();
      await refreshComparison();

      loadingState.classList.add("hidden");
      errorState.classList.add("hidden");
      compareLayout.classList.remove("hidden");
    } catch (error) {
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      compareLayout.classList.add("hidden");
      pageTitle.textContent = "Comparison unavailable";
      pageLead.textContent = "Could not load comparison data for this track.";
    }
  }

  window.addEventListener("resize", requestSyncCompareHeights);

  audioPlayer.addEventListener("loadedmetadata", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("play", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("timeupdate", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("seeking", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("seeked", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("pause", updateSpectrogramPlayhead);
  audioPlayer.addEventListener("ended", () => {
    spectrogramPlayhead.style.left = "100%";
    spectrogramPlayhead.classList.remove("hidden");
  });
  audioPlayer.addEventListener("emptied", resetSpectrogramPlayhead);

  leftModelSelect.addEventListener("change", async () => {
    leftModel = leftModelSelect.value;
    await refreshComparison();
  });

  rightModelSelect.addEventListener("change", async () => {
    rightModel = rightModelSelect.value;
    await refreshComparison();
  });

  loadComparePage();
});
