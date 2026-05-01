document.addEventListener("DOMContentLoaded", () => {
  const trackId = searchParams().get("track_id");
  let model = searchParams().get("model") || "panns";

  const pageTitle = document.getElementById("pageTitle");
  const pageLead = document.getElementById("pageLead");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const songLayout = document.getElementById("songLayout");
  const entityLinks = document.getElementById("entityLinks");
  const songTitle = document.getElementById("songTitle");
  const songArtist = document.getElementById("songArtist");
  const songGenres = document.getElementById("songGenres");
  const spectrogramImage = document.getElementById("spectrogramImage");
  const spectrogramPlayhead = document.getElementById("spectrogramPlayhead");
  const audioPlayer = document.getElementById("audioPlayer");
  const metaGrid = document.getElementById("metaGrid");
  const similarList = document.getElementById("similarList");
  const modelSelect = document.getElementById("modelSelect");
  const compareButton = document.getElementById("compareButton");

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

  function updateUrlModel(nextModel) {
    const url = new URL(window.location.href);
    url.searchParams.set("model", nextModel);
    history.replaceState({}, "", url);
  }

  async function loadAvailableModels() {
    const rootData = await loadRootInfo();
    const availableModels = Array.isArray(rootData.available_models)
      ? rootData.available_models
      : ["panns"];

    const defaultModel = rootData.default_model || availableModels[0] || "panns";

    if (!availableModels.includes(model)) {
      model = defaultModel;
    }

    modelSelect.innerHTML = availableModels.map((modelName) => `
      <option value="${escapeHtml(modelName)}" ${modelName === model ? "selected" : ""}>
        ${escapeHtml(modelName.toUpperCase())}
      </option>
    `).join("");

    updateUrlModel(model);
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

  function renderSimilarCard(track, index) {
    const t = normalizeTrack(track);
    const rankText = `#${index + 1}`;
    const scoreText = Number.isFinite(Number(t.score))
      ? `${(Number(t.score) * 100).toFixed(2)}% similar`
      : "";

    return `
      <article class="card card-tight">
        <div class="similar-topline">
          <a class="card-title" href="song.html?track_id=${encodeURIComponent(t.track_id)}&model=${encodeURIComponent(model)}">
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

  async function loadSongPage() {
    if (!trackId) {
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      errorState.textContent = "Missing track_id parameter.";
      pageTitle.textContent = "Track not specified";
      pageLead.textContent = "Open this page from search, artist, album, or genre pages.";
      return;
    }

    try {
      await loadAvailableModels();

      const data = await fetchJson(`${SITE_CONFIG.API_BASE}/tracks/${encodeURIComponent(trackId)}?model=${encodeURIComponent(model)}`);
      const track = normalizeTrack(data);
      compareButton.href = `compare.html?track_id=${encodeURIComponent(track.track_id)}`;

      const rawSimilar = Array.isArray(
        data.recommendations || data.similar_tracks || data.results || data.neighbors
      )
        ? (data.recommendations || data.similar_tracks || data.results || data.neighbors).map(normalizeTrack)
        : [];

      const seen = new Set();
      const similar = rawSimilar.filter((item) => {
        const artist = (item.artist || "").toLowerCase().trim();
        const title = (item.title || "").toLowerCase().trim();
        const key = `${artist}::${title}`;

        if (
          artist === (track.artist || "").toLowerCase().trim() &&
          title === (track.title || "").toLowerCase().trim()
        ) {
          return false;
        }

        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

      pageTitle.textContent = track.title;
      pageLead.textContent = `Similarity model: ${model.toUpperCase()}`;
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

      resetSpectrogramPlayhead();

      metaGrid.innerHTML = "";
      addMeta("Track ID", escapeHtml(track.track_id));
      addMeta("Title", escapeHtml(track.title));
      addMeta("Artist", `<a href="artist.html?name=${encodeURIComponent(track.artist)}">${escapeHtml(track.artist)}</a>`);
      addMeta("Album", track.album ? `<a href="album.html?name=${encodeURIComponent(track.album)}">${escapeHtml(track.album)}</a>` : "Unknown album");
      addMeta("Duration", escapeHtml(formatDuration(track.duration) || track.duration || "Unknown"));
      addMeta("License", track.license_title ? linkOrText(track.license_url, track.license_title) : "Unknown");
      addMeta("Artist Website", track.artist_website ? linkOrText(track.artist_website, track.artist_website) : "Not provided");

      similarList.innerHTML = similar.length
        ? similar.map((item, index) => renderSimilarCard(item, index)).join("")
        : `<div class="card-meta">No similar tracks were returned for this song.</div>`;

      loadingState.classList.add("hidden");
      errorState.classList.add("hidden");
      songLayout.classList.remove("hidden");
    } catch (error) {
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      songLayout.classList.add("hidden");
      pageTitle.textContent = "Track unavailable";
      pageLead.textContent = "No track data could be loaded for this page.";
    }
  }

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

  modelSelect.addEventListener("change", async () => {
    model = modelSelect.value;
    updateUrlModel(model);
    await loadSongPage();
  });

  loadSongPage();
});
