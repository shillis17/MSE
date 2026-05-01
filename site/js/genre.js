document.addEventListener("DOMContentLoaded", () => {
  const params = searchParams();
  const genreName = params.get("name") || params.get("genre");

  const pageTitle = document.getElementById("pageTitle");
  const pageLead = document.getElementById("pageLead");
  const trackFilter = document.getElementById("trackFilter");
  const sortSelect = document.getElementById("sortSelect");
  const toolbarMeta = document.getElementById("toolbarMeta");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const emptyState = document.getElementById("emptyState");
  const songsGrid = document.getElementById("songsGrid");
  const prevButton = document.getElementById("prevButton");
  const nextButton = document.getElementById("nextButton");

  let currentPage = 1;
  let totalPages = 1;
  let detailController = null;

  async function loadGenreSongs(page = 1) {
    if (!genreName) {
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
      errorState.textContent = "Missing genre name parameter.";
      pageTitle.textContent = "Genre not specified";
      pageLead.textContent = "Open this page from the genre browser.";
      return;
    }

    if (detailController) detailController.abort();
    detailController = new AbortController();

    loadingState.classList.remove("hidden");
    errorState.classList.add("hidden");
    emptyState.classList.add("hidden");
    songsGrid.innerHTML = "";

    try {
      const url = new URL(`${SITE_CONFIG.API_BASE}/tracks`);
      url.searchParams.set("genre", genreName);
      url.searchParams.set("page", page);
      url.searchParams.set("limit", 20);
      url.searchParams.set("sort", sortSelect.value);
      if (trackFilter.value.trim()) url.searchParams.set("search", trackFilter.value.trim());

      const data = await fetchJson(url.toString(), { signal: detailController.signal });
      const tracks = Array.isArray(data.tracks) ? data.tracks : [];

      currentPage = Number(data.page) || 1;
      totalPages = Number(data.total_pages) || 1;

      pageTitle.textContent = genreName;
      pageLead.textContent = `${Number(data.total_results || tracks.length).toLocaleString()} matching track(s)`;
      toolbarMeta.textContent = `Page ${currentPage} of ${totalPages}`;
      prevButton.disabled = currentPage <= 1;
      nextButton.disabled = currentPage >= totalPages;

      loadingState.classList.add("hidden");

      if (tracks.length === 0) {
        emptyState.classList.remove("hidden");
        return;
      }

      songsGrid.innerHTML = tracks.map(buildTrackCard).join("");
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error(error);
      loadingState.classList.add("hidden");
      errorState.classList.remove("hidden");
    }
  }

  trackFilter.addEventListener("input", debounce(() => loadGenreSongs(1), 250));
  sortSelect.addEventListener("change", () => loadGenreSongs(1));
  prevButton.addEventListener("click", () => currentPage > 1 && loadGenreSongs(currentPage - 1));
  nextButton.addEventListener("click", () => currentPage < totalPages && loadGenreSongs(currentPage + 1));

  loadGenreSongs();
});
