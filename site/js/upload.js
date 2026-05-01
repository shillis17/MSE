document.addEventListener("DOMContentLoaded", () => {
  const CLIP_DURATION = 30;

  const audioFileInput = document.getElementById("audioFile");
  const previewPlayer = document.getElementById("previewPlayer");
  const startSlider = document.getElementById("startSlider");
  const timeReadout = document.getElementById("timeReadout");
  const searchButton = document.getElementById("searchButton");
  const statusText = document.getElementById("statusText");
  const warningText = document.getElementById("warningText");
  const healthText = document.getElementById("healthText");

  const queryPanel = document.getElementById("queryPanel");
  const queryFileName = document.getElementById("queryFileName");
  const querySegment = document.getElementById("querySegment");

  const resultsPanel = document.getElementById("resultsPanel");
  const pannsMeta = document.getElementById("pannsMeta");
  const clapMeta = document.getElementById("clapMeta");
  const pannsList = document.getElementById("pannsList");
  const clapList = document.getElementById("clapList");

  let selectedFile = null;
  let previewUrl = null;
  let audioDuration = 0;

  function secondsToClock(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins}:${String(secs).padStart(2, "0")}`;
  }

  function updateTimeReadout() {
    const start = Number(startSlider.value || 0);
    const end = Math.min(start + CLIP_DURATION, audioDuration || CLIP_DURATION);
    timeReadout.textContent = `Start: ${secondsToClock(start)} · End: ${secondsToClock(end)}`;
  }

  function clearResults() {
    resultsPanel.classList.add("hidden");
    pannsList.innerHTML = "";
    clapList.innerHTML = "";
    pannsMeta.textContent = "";
    clapMeta.textContent = "";
  }

  function renderUploadResultCard(track, index, modelName) {
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

  function renderColumn(targetEl, metaEl, modelName, results) {
    metaEl.textContent = `${results.length} result(s)`;

    targetEl.innerHTML = results.length
      ? results.map((track, index) => renderUploadResultCard(track, index, modelName)).join("")
      : `<div class="card-meta">No results returned.</div>`;
  }

  async function loadHealth() {
    try {
      const data = await fetchJson(`${SITE_CONFIG.API_BASE}/health`);
      healthText.textContent = `Backend status: ${String(data.status || "unknown")} · upload assets ready: ${String(Boolean(data.upload_assets_ready))} · device: ${String(data.device || "unknown")}`;
    } catch (error) {
      console.error(error);
      healthText.textContent = "Backend health could not be loaded.";
    }
  }

  audioFileInput.addEventListener("change", () => {
    clearResults();
    warningText.classList.add("hidden");
    warningText.textContent = "";

    const file = audioFileInput.files?.[0];
    selectedFile = file || null;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }

    if (!selectedFile) {
      previewPlayer.classList.add("hidden");
      previewPlayer.removeAttribute("src");
      startSlider.disabled = true;
      searchButton.disabled = true;
      statusText.textContent = "Choose a file to begin.";
      return;
    }

    previewUrl = URL.createObjectURL(selectedFile);
    previewPlayer.src = previewUrl;
    previewPlayer.classList.remove("hidden");
    statusText.textContent = "Loading audio metadata...";
    searchButton.disabled = true;
  });

  previewPlayer.addEventListener("loadedmetadata", () => {
    audioDuration = Number(previewPlayer.duration || 0);

    if (!Number.isFinite(audioDuration) || audioDuration <= 0) {
      statusText.textContent = "Could not read audio duration.";
      return;
    }

    const maxStart = Math.max(0, audioDuration - CLIP_DURATION);
    startSlider.max = String(maxStart);
    startSlider.value = "0";
    startSlider.disabled = false;
    searchButton.disabled = false;

    updateTimeReadout();

    statusText.textContent = audioDuration < CLIP_DURATION
      ? "This file is shorter than 30 seconds. The backend will pad the clip as needed."
      : "Pick a 30-second segment, then search.";
  });

  startSlider.addEventListener("input", updateTimeReadout);

  searchButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    clearResults();

    const startSeconds = Number(startSlider.value || 0);

    queryPanel.classList.remove("hidden");
    queryFileName.textContent = `File: ${selectedFile.name}`;
    querySegment.textContent = `Segment: ${secondsToClock(startSeconds)} to ${secondsToClock(startSeconds + CLIP_DURATION)}`;

    statusText.textContent = "Uploading clip and searching...";
    searchButton.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("start_seconds", String(startSeconds));
      formData.append("duration_seconds", String(CLIP_DURATION));

      const response = await fetch(`${SITE_CONFIG.API_BASE}/upload-search`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Upload search failed with status ${response.status}`);
      }

      const data = await response.json();

      const pannsResults = Array.isArray(data.results?.panns) ? data.results.panns.map(normalizeTrack) : [];
      const clapResults = Array.isArray(data.results?.clap) ? data.results.clap.map(normalizeTrack) : [];

      renderColumn(pannsList, pannsMeta, "panns", pannsResults);
      renderColumn(clapList, clapMeta, "clap", clapResults);

      resultsPanel.classList.remove("hidden");
      statusText.textContent = "Done.";
    } catch (error) {
      console.error(error);
      warningText.textContent = "Search failed. Check the backend logs and confirm that upload-search assets are ready.";
      warningText.classList.remove("hidden");
      statusText.textContent = "Search failed.";
    } finally {
      searchButton.disabled = false;
    }
  });

  loadHealth();
});
