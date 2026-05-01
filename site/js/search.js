document.addEventListener("DOMContentLoaded", () => {
  const PAGE_LIMIT = 20;

  const searchInput = document.getElementById("searchInput");
  const sortSelect = document.getElementById("sortSelect");
  const searchMeta = document.getElementById("searchMeta");
  const resultsShell = document.getElementById("resultsShell");
  const resultsHeader = document.getElementById("resultsHeader");
  const loadingState = document.getElementById("loadingState");
  const emptyState = document.getElementById("emptyState");
  const resultsGrid = document.getElementById("resultsGrid");
  const pagination = document.getElementById("pagination");
  const prevButton = document.getElementById("prevButton");
  const nextButton = document.getElementById("nextButton");

  let currentQuery = "";
  let currentPage = 1;
  let hasNextPage = false;
  let hasPrevPage = false;
  let searchController = null;

  function clearResults() {
    resultsGrid.innerHTML = "";
    resultsHeader.classList.add("hidden");
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    pagination.classList.add("hidden");
    prevButton.disabled = true;
    nextButton.disabled = true;
  }

  async function runSearch(query, page = 1) {
    const trimmed = query.trim();

    if (trimmed.length < 2) {
      currentQuery = "";
      currentPage = 1;
      searchMeta.textContent = "Type at least 2 characters to search.";
      resultsShell.classList.add("hidden");
      clearResults();
      return;
    }

    if (searchController) searchController.abort();
    searchController = new AbortController();

    resultsShell.classList.remove("hidden");
    loadingState.classList.remove("hidden");
    emptyState.classList.add("hidden");
    resultsHeader.classList.add("hidden");
    pagination.classList.add("hidden");
    resultsGrid.innerHTML = "";
    searchMeta.textContent = `Searching for "${trimmed}"...`;

    try {
      const url = new URL(`${SITE_CONFIG.API_BASE}/tracks`);
      url.searchParams.set("search", trimmed);
      url.searchParams.set("page", page);
      url.searchParams.set("limit", PAGE_LIMIT);
      url.searchParams.set("sort", sortSelect.value);

      const data = await fetchJson(url.toString(), { signal: searchController.signal });
      const tracks = Array.isArray(data.tracks) ? data.tracks : [];

      currentQuery = trimmed;
      currentPage = Number(data.page) || 1;
      hasNextPage = Boolean(data.has_next);
      hasPrevPage = Boolean(data.has_prev);

      loadingState.classList.add("hidden");

      if (tracks.length === 0) {
        clearResults();
        resultsShell.classList.remove("hidden");
        emptyState.classList.remove("hidden");
        searchMeta.textContent = "No matching tracks found.";
        return;
      }

      resultsGrid.innerHTML = tracks.map(buildTrackCard).join("");
      resultsHeader.textContent = `${Number(data.total_results || tracks.length).toLocaleString()} result(s)`;
      resultsHeader.classList.remove("hidden");

      prevButton.disabled = !hasPrevPage;
      nextButton.disabled = !hasNextPage;
      pagination.classList.remove("hidden");
      searchMeta.textContent = `Showing page ${currentPage} of ${Number(data.total_pages || 1)}.`;
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error(error);
      clearResults();
      resultsShell.classList.remove("hidden");
      emptyState.classList.remove("hidden");
      emptyState.textContent = "Search failed.";
      searchMeta.textContent = "The request could not be completed.";
    }
  }

  const debouncedSearch = debounce(() => runSearch(searchInput.value, 1), 250);

  searchInput.addEventListener("input", debouncedSearch);
  sortSelect.addEventListener("change", () => {
    if (searchInput.value.trim().length >= 2) {
      runSearch(searchInput.value, 1);
    }
  });

  prevButton.addEventListener("click", () => {
    if (hasPrevPage) runSearch(currentQuery, currentPage - 1);
  });

  nextButton.addEventListener("click", () => {
    if (hasNextPage) runSearch(currentQuery, currentPage + 1);
  });
});
