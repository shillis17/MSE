document.addEventListener("DOMContentLoaded", () => {
  const artistName = searchParams().get("name");
  const pageTitle = document.getElementById("pageTitle");
  const pageLead = document.getElementById("pageLead");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const pageLayout = document.getElementById("pageLayout");
  const artistMeta = document.getElementById("artistMeta");
  const albumsList = document.getElementById("albumsList");
  const tracksList = document.getElementById("tracksList");

  function addMeta(label, value) {
    const item = document.createElement("div");
    item.className = "meta-item";
    item.innerHTML = `<span class="meta-label">${label}</span><div class="meta-value">${value}</div>`;
    artistMeta.appendChild(item);
  }

  function renderArtistMeta(name, tracks, albums) {
    artistMeta.innerHTML = "";
    const uniqueGenres = [...new Set(
      tracks.flatMap((track) => Array.isArray(track.genres) ? track.genres : [])
    )].sort((a, b) => String(a).localeCompare(String(b)));

    addMeta("Artist Name", escapeHtml(name || "Unknown artist"));
    addMeta("Track Count", escapeHtml(tracks.length));
    addMeta("Album Count", escapeHtml(albums.length));
    addMeta(
      "Genres",
      uniqueGenres.length
        ? uniqueGenres.map((genre) => `<a href="genre.html?name=${encodeURIComponent(genre)}">${escapeHtml(genre)}</a>`).join(", ")
        : "No Genre"
    );
  }

  function renderAlbums(albums) {
    if (!albums.length) {
      albumsList.innerHTML = `<div class="card-meta">No albums found.</div>`;
      return;
    }

    albumsList.innerHTML = albums.map((albumName) => `
      <article class="card card-tight">
        <a class="card-title" href="album.html?name=${encodeURIComponent(albumName)}">${escapeHtml(albumName || "Untitled album")}</a>
      </article>
    `).join("");
  }

  function renderTracks(tracks) {
    tracksList.innerHTML = tracks.length
      ? tracks.map(buildTrackCard).join("")
      : `<div class="card-meta">No tracks found.</div>`;
  }

  async function loadArtistPage() {
    if (!artistName) {
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      errorState.textContent = "Missing artist name parameter.";
      pageTitle.textContent = "Artist not specified";
      pageLead.textContent = "Open this page from the artist browser.";
      return;
    }

    try {
      const data = await fetchJson(`${SITE_CONFIG.API_BASE}/artists/${encodeURIComponent(artistName)}`);
      const tracks = Array.isArray(data.tracks) ? data.tracks.map(normalizeTrack) : [];
      const albums = Array.isArray(data.albums) ? data.albums : [];

      if (!tracks.length) throw new Error("No tracks found for artist");

      pageTitle.textContent = artistName;
      pageLead.textContent = `${tracks.length} track(s) across ${albums.length} album(s)`;
      renderArtistMeta(artistName, tracks, albums);
      renderAlbums(albums);
      renderTracks(tracks);

      loadingState.classList.add("hidden");
      errorState.classList.add("hidden");
      pageLayout.classList.remove("hidden");
    } catch (error) {
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      pageLayout.classList.add("hidden");
      pageTitle.textContent = artistName || "Artist unavailable";
      pageLead.textContent = "No artist data could be loaded for this page.";
    }
  }

  loadArtistPage();
});
