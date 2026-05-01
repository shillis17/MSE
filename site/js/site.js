const SITE_CONFIG = {
  API_BASE: "https://mu-9273f5d2d4b64326910d11d79bc6a898.ecs.us-east-1.on.aws",
  
  //API_BASE: "http://127.0.0.1:8000"
};

const NAV_ITEMS = [
  ["home", "Home", "home.html"],
  ["search", "Search", "search.html"],
  ["genres", "Genres", "genres.html"],
  ["artists", "Artists", "artists.html"],
  ["albums", "Albums", "albums.html"],
  ["uploads", "Uploads", "upload.html"]
];

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function debounce(fn, wait = 250) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function parseDurationToSeconds(duration) {
  if (duration == null || duration === "") return null;
  if (typeof duration === "number" && Number.isFinite(duration)) return duration;

  const text = String(duration).trim();
  if (!text) return null;

  const numeric = Number(text);
  if (Number.isFinite(numeric)) return numeric;

  const parts = text.split(":").map((part) => Number(part.trim()));
  if (parts.some((part) => !Number.isFinite(part))) return null;

  if (parts.length === 2) {
    const [mins, secs] = parts;
    return (mins * 60) + secs;
  }

  if (parts.length === 3) {
    const [hours, mins, secs] = parts;
    return (hours * 3600) + (mins * 60) + secs;
  }

  return null;
}

function formatDuration(duration) {
  const totalSeconds = parseDurationToSeconds(duration);
  if (totalSeconds == null) return "";

  const hours = Math.floor(totalSeconds / 3600);
  const mins = Math.floor((totalSeconds % 3600) / 60);
  const secs = Math.round(totalSeconds % 60);

  if (hours > 0) {
    return `${hours}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function normalizeTrack(track = {}) {
  return {
    track_id: track.track_id,
    title: track.title || track.track_title || "Untitled",
    artist: track.artist || track.artist_name || "Unknown artist",
    album: track.album || track.album_title || "Unknown album",
    duration: track.duration || track.track_duration || "",
    genres: Array.isArray(track.genres || track.track_genres)
      ? (track.genres || track.track_genres)
      : [],
    rank: track.rank,
    score: track.score,
    spectrogram_url: track.spectrogram_url,
    audio_url: track.audio_url,
    license_title: track.license_title,
    license_url: track.license_url,
    artist_website: track.artist_website,
    track_url: track.track_url
  };
}

function renderGenres(genres = []) {
  if (!Array.isArray(genres) || genres.length === 0) {
    return '<span class="genre-chip">No Genre</span>';
  }

  return genres
    .map((genre) => `<a class="genre-chip" href="genre.html?name=${encodeURIComponent(genre)}">${escapeHtml(genre)}</a>`)
    .join("");
}

function buildTrackCard(track) {
  const t = normalizeTrack(track);
  const artistLink = `<a href="artist.html?name=${encodeURIComponent(t.artist)}">${escapeHtml(t.artist)}</a>`;
  const albumLink = t.album ? ` · <a href="album.html?name=${encodeURIComponent(t.album)}">${escapeHtml(t.album)}</a>` : "";
  const duration = t.duration ? ` · ${escapeHtml(formatDuration(t.duration) || t.duration)}` : "";

  return `
    <article class="card card-tight">
      <a class="card-title" href="song.html?track_id=${encodeURIComponent(t.track_id)}">${escapeHtml(t.title)}</a>
      <div class="card-meta">${artistLink}${albumLink}${duration}</div>
      <div class="genre-list">${renderGenres(t.genres)}</div>
    </article>
  `;
}

function mountSiteChrome() {
  const page = document.body.dataset.page || "";
  const headerMount = document.getElementById("siteHeader");
  const footerMount = document.getElementById("siteFooter");

  if (headerMount) {
    headerMount.innerHTML = `
      <header class="site-header">
        <div class="container site-nav">
          <a class="brand" href="home.html">Music Similarity Explorer</a>
          <nav class="nav-links">
            ${NAV_ITEMS.map(([key, label, href]) => `
              <a class="nav-link ${page === key ? "active" : ""}" href="${href}">${label}</a>
            `).join("")}
          </nav>
        </div>
      </header>
    `;
  }

  if (footerMount) {
    footerMount.innerHTML = `
      <footer class="footer">
        <div class="container">
          <p>Music Similarity Explorer · Static frontend with ECS backend</p>
          <p class="footer-copy footer-copy-primary">
            This project uses audio and metadata from the Free Music Archive (FMA).
            All tracks remain the property of their respective artists and are used under their original Creative Commons licenses.
          </p>
          <p class="footer-copy footer-copy-secondary">
            Attribution — Appropriate credit is provided on each track page, including artist name, license type, and a link to the license.
            Audio tracks are presented as 30-second excerpts of the original works; no endorsement by the original creators is implied.
          </p>
          <p class="footer-copy footer-copy-secondary">
            Additional license terms may apply depending on the track, including NonCommercial (NC), NoDerivatives (ND), or ShareAlike (SA) conditions.
            Users are responsible for complying with individual track licenses.
          </p>
        </div>
      </footer>
    `;
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const text = await response.text();
      if (text) message = text;
    } catch (_) {
      // ignore body parse errors
    }
    throw new Error(message);
  }
  return response.json();
}

function searchParams() {
  return new URLSearchParams(window.location.search);
}

async function loadRootInfo() {
  return fetchJson(`${SITE_CONFIG.API_BASE}/`);
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

document.addEventListener("DOMContentLoaded", mountSiteChrome);
