document.addEventListener("DOMContentLoaded", () => {
  const albumName = searchParams().get("name");
  const pageTitle = document.getElementById("pageTitle");
  const pageLead = document.getElementById("pageLead");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const pageLayout = document.getElementById("pageLayout");
  const albumMeta = document.getElementById("albumMeta");
  const artistsList = document.getElementById("artistsList");
  const tracksList = document.getElementById("tracksList");

  function addMeta(label, value) {
    const item = document.createElement("div");
    item.className = "meta-item";
    item.innerHTML = `<span class="meta-label">${label}</span><div class="meta-value">${value}</div>`;
    albumMeta.appendChild(item);
  }

  function renderAlbumMeta(name, tracks) {
    albumMeta.innerHTML = "";
    const uniqueArtists = [...new Set(tracks.map((track) => track.artist).filter(Boolean))]
      .sort((a, b) => String(a).localeCompare(String(b)));
    const uniqueGenres = [...new Set(
      tracks.flatMap((track) => Array.isArray(track.genres) ? track.genres : [])
    )].sort((a, b) => String(a).localeCompare(String(b)));
    const totalSeconds = tracks.reduce((sum, track) => sum + (parseDurationToSeconds(track.duration) || 0), 0);

    addMeta("Album Name", escapeHtml(name || "Unknown album"));
    addMeta("Track Count", escapeHtml(tracks.length));
    addMeta("Artist Count", escapeHtml(uniqueArtists.length));
    addMeta("Total Length", escapeHtml(totalSeconds ? formatDuration(totalSeconds) : "Unknown"));
    addMeta(
      "Artists",
      uniqueArtists.length
        ? uniqueArtists.map((artist) => `<a href="artist.html?name=${encodeURIComponent(artist)}">${escapeHtml(artist)}</a>`).join(", ")
        : "Unknown Artist"
    );
    addMeta(
      "Genres",
      uniqueGenres.length
        ? uniqueGenres.map((genre) => `<a href="genre.html?name=${encodeURIComponent(genre)}">${escapeHtml(genre)}</a>`).join(", ")
        : "No Genre"
    );
  }

  function renderArtists(artists) {
    artistsList.innerHTML = artists.length
      ? artists.map((artistName) => `
          <article class="card card-tight">
            <a class="card-title" href="artist.html?name=${encodeURIComponent(artistName)}">${escapeHtml(artistName || "Unknown artist")}</a>
          </article>
        `).join("")
      : `<div class="card-meta">No artists found.</div>`;
  }

  function renderTracks(tracks) {
    tracksList.innerHTML = tracks.length
      ? tracks.map(buildTrackCard).join("")
      : `<div class="card-meta">No tracks found.</div>`;
  }

  async function loadAlbumPage() {
    if (!albumName) {
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      errorState.textContent = "Missing album name parameter.";
      pageTitle.textContent = "Album not specified";
      pageLead.textContent = "Open this page from the album browser.";
      return;
    }

    try {
      const data = await fetchJson(`${SITE_CONFIG.API_BASE}/albums/${encodeURIComponent(albumName)}`);
      const tracks = Array.isArray(data.tracks) ? data.tracks.map(normalizeTrack) : [];
      const artists = Array.isArray(data.artists) ? data.artists : [];

      if (!tracks.length) throw new Error("No tracks found for album");

      pageTitle.textContent = albumName;
      pageLead.textContent = `${tracks.length} track(s) by ${artists.length} artist(s)`;
      renderAlbumMeta(albumName, tracks);
      renderArtists(artists);
      renderTracks(tracks);

      loadingState.classList.add("hidden");
      errorState.classList.add("hidden");
      pageLayout.classList.remove("hidden");
    } catch (error) {
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      pageLayout.classList.add("hidden");
      pageTitle.textContent = albumName || "Album unavailable";
      pageLead.textContent = "No album data could be loaded for this page.";
    }
  }

  loadAlbumPage();
});
