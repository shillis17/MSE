document.addEventListener("DOMContentLoaded", () => {
  const PAGE_LIMIT = 60;
  const genreFilter = document.getElementById("genreFilter");
  const sortSelect = document.getElementById("sortSelect");
  const genreMeta = document.getElementById("genreMeta");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const emptyState = document.getElementById("emptyState");
  const genresGrid = document.getElementById("genresGrid");
  const prevButton = document.getElementById("prevButton");
  const nextButton = document.getElementById("nextButton");

  let currentPage = 1;
  let totalPages = 1;
  let listController = null;

  function renderGenreCard(item) {
    return `
      <article class="card card-tight">
        <a class="card-title" href="genre.html?name=${encodeURIComponent(item.name)}">${escapeHtml(item.label || item.name)}</a>
        <div class="card-meta">${Number(item.track_count ?? item.count ?? 0).toLocaleString()} track(s)</div>
      </article>
    `;
  }

  async function loadGenres(page = 1) {
    if (listController) listController.abort();
    listController = new AbortController();

    loadingState.classList.remove("hidden");
    errorState.classList.add("hidden");
    emptyState.classList.add("hidden");
    genresGrid.innerHTML = "";

    try {
      const url = new URL(`${SITE_CONFIG.API_BASE}/genres`);
      url.searchParams.set("page", page);
      url.searchParams.set("limit", PAGE_LIMIT);
      url.searchParams.set("sort", sortSelect.value);
      if (genreFilter.value.trim()) url.searchParams.set("search", genreFilter.value.trim());

      const data = await fetchJson(url.toString(), { signal: listController.signal });
      const items = Array.isArray(data.genres) ? data.genres : [];

      currentPage = Number(data.page) || 1;
      totalPages = Number(data.total_pages) || 1;

      loadingState.classList.add("hidden");
      prevButton.disabled = currentPage <= 1;
      nextButton.disabled = currentPage >= totalPages;
      genreMeta.textContent = `${Number(data.total_results || items.length).toLocaleString()} genre(s) · page ${currentPage} of ${totalPages}`;

      if (items.length === 0) {
        emptyState.classList.remove("hidden");
        return;
      }

      genresGrid.innerHTML = items.map(renderGenreCard).join("");
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
    }
  }

  const debouncedLoadGenres = debounce(() => loadGenres(1), 250);
  genreFilter.addEventListener("input", debouncedLoadGenres);
  sortSelect.addEventListener("change", () => loadGenres(1));
  prevButton.addEventListener("click", () => currentPage > 1 && loadGenres(currentPage - 1));
  nextButton.addEventListener("click", () => currentPage < totalPages && loadGenres(currentPage + 1));

  loadGenres();
});
