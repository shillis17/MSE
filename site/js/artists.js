document.addEventListener("DOMContentLoaded", () => {
  const PAGE_LIMIT = 60;
  const artistFilter = document.getElementById("artistFilter");
  const sortSelect = document.getElementById("sortSelect");
  const artistMeta = document.getElementById("artistMeta");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const emptyState = document.getElementById("emptyState");
  const artistsGrid = document.getElementById("artistsGrid");
  const prevButton = document.getElementById("prevButton");
  const nextButton = document.getElementById("nextButton");

  let currentPage = 1;
  let totalPages = 1;
  let listController = null;

  function renderArtistCard(item) {
    return `
      <article class="card card-tight">
        <a class="card-title" href="artist.html?name=${encodeURIComponent(item.name)}">${escapeHtml(item.name)}</a>
        <div class="card-meta">${Number(item.track_count ?? item.count ?? 0).toLocaleString()} track(s)</div>
      </article>
    `;
  }

  async function loadArtists(page = 1) {
    if (listController) listController.abort();
    listController = new AbortController();

    loadingState.classList.remove("hidden");
    errorState.classList.add("hidden");
    emptyState.classList.add("hidden");
    artistsGrid.innerHTML = "";

    try {
      const url = new URL(`${SITE_CONFIG.API_BASE}/artists`);
      url.searchParams.set("page", page);
      url.searchParams.set("limit", PAGE_LIMIT);
      url.searchParams.set("sort", sortSelect.value);
      if (artistFilter.value.trim()) url.searchParams.set("search", artistFilter.value.trim());

      const data = await fetchJson(url.toString(), { signal: listController.signal });
      const items = Array.isArray(data.artists) ? data.artists : [];

      currentPage = Number(data.page) || 1;
      totalPages = Number(data.total_pages) || 1;

      loadingState.classList.add("hidden");
      prevButton.disabled = currentPage <= 1;
      nextButton.disabled = currentPage >= totalPages;
      artistMeta.textContent = `${Number(data.total_results || items.length).toLocaleString()} artist(s) · page ${currentPage} of ${totalPages}`;

      if (items.length === 0) {
        emptyState.classList.remove("hidden");
        return;
      }

      artistsGrid.innerHTML = items.map(renderArtistCard).join("");
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
    }
  }

  const debouncedLoadArtists = debounce(() => loadArtists(1), 250);
  artistFilter.addEventListener("input", debouncedLoadArtists);
  sortSelect.addEventListener("change", () => loadArtists(1));
  prevButton.addEventListener("click", () => currentPage > 1 && loadArtists(currentPage - 1));
  nextButton.addEventListener("click", () => currentPage < totalPages && loadArtists(currentPage + 1));

  loadArtists();
});
