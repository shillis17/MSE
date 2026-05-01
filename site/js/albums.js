document.addEventListener("DOMContentLoaded", () => {
  const PAGE_LIMIT = 60;
  const albumFilter = document.getElementById("albumFilter");
  const sortSelect = document.getElementById("sortSelect");
  const albumMeta = document.getElementById("albumMeta");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const emptyState = document.getElementById("emptyState");
  const albumsGrid = document.getElementById("albumsGrid");
  const prevButton = document.getElementById("prevButton");
  const nextButton = document.getElementById("nextButton");

  let currentPage = 1;
  let totalPages = 1;
  let listController = null;

  function renderAlbumCard(item) {
    return `
      <article class="card card-tight">
        <a class="card-title" href="album.html?name=${encodeURIComponent(item.name)}">${escapeHtml(item.name)}</a>
        <div class="card-meta">${Number(item.track_count ?? item.count ?? 0).toLocaleString()} track(s)</div>
      </article>
    `;
  }

  async function loadAlbums(page = 1) {
    if (listController) listController.abort();
    listController = new AbortController();

    loadingState.classList.remove("hidden");
    errorState.classList.add("hidden");
    emptyState.classList.add("hidden");
    albumsGrid.innerHTML = "";

    try {
      const url = new URL(`${SITE_CONFIG.API_BASE}/albums`);
      url.searchParams.set("page", page);
      url.searchParams.set("limit", PAGE_LIMIT);
      url.searchParams.set("sort", sortSelect.value);
      if (albumFilter.value.trim()) url.searchParams.set("search", albumFilter.value.trim());

      const data = await fetchJson(url.toString(), { signal: listController.signal });
      const items = Array.isArray(data.albums) ? data.albums : [];

      currentPage = Number(data.page) || 1;
      totalPages = Number(data.total_pages) || 1;

      loadingState.classList.add("hidden");
      prevButton.disabled = currentPage <= 1;
      nextButton.disabled = currentPage >= totalPages;
      albumMeta.textContent = `${Number(data.total_results || items.length).toLocaleString()} album(s) · page ${currentPage} of ${totalPages}`;

      if (items.length === 0) {
        emptyState.classList.remove("hidden");
        return;
      }

      albumsGrid.innerHTML = items.map(renderAlbumCard).join("");
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
    }
  }

  const debouncedLoadAlbums = debounce(() => loadAlbums(1), 250);
  albumFilter.addEventListener("input", debouncedLoadAlbums);
  sortSelect.addEventListener("change", () => loadAlbums(1));
  prevButton.addEventListener("click", () => currentPage > 1 && loadAlbums(currentPage - 1));
  nextButton.addEventListener("click", () => currentPage < totalPages && loadAlbums(currentPage + 1));

  loadAlbums();
});
