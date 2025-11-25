const API_BASE = "http://127.0.0.1:8000";

function getQueryTitle() {
  const params = new URLSearchParams(window.location.search);
  const title = params.get("title");
  return title ? title.trim() : null;
}

async function loadMoviePage() {
  const title = getQueryTitle();

  // Elements DOM de movie.html
  const pageTitleEl = document.getElementById("movie-page-title");
  const posterLargeEl = document.getElementById("movie-poster-large");
  const genresEl = document.getElementById("movie-genres");
  const directorEl = document.getElementById("movie-director");
  const castEl = document.getElementById("movie-cast");
  const taglineEl = document.getElementById("movie-tagline");
  const overviewEl = document.getElementById("movie-overview");
  const similarContainer = document.getElementById("similar-results");

  if (!title) {
    console.error("Aucun titre dans l’URL (movie.html?title=...)");
    return;
  }

  // Titre en haut de page
  if (pageTitleEl) pageTitleEl.textContent = title;

  try {
    // 1) Détails du film
    const res = await fetch(
      `${API_BASE}/movie?title=${encodeURIComponent(title)}`
    );
    if (!res.ok) {
      console.error("Film introuvable", res.status);
      return;
    }
    const movie = await res.json();

    if (pageTitleEl) pageTitleEl.textContent = movie.title || title;

    if (genresEl) {
      genresEl.textContent = movie.genres || "";
    }

    if (directorEl) {
      directorEl.textContent = movie.director
        ? `Réalisé par ${movie.director}`
        : "";
    }

    if (castEl) {
      castEl.textContent = movie.cast
        ? `Avec ${movie.cast}`
        : "";
    }

    if (taglineEl) {
      taglineEl.textContent =
        movie.tagline || "Aucune tagline disponible.";
    }

    if (overviewEl) {
      overviewEl.textContent =
        movie.overview || "Aucun résumé disponible.";
    }

    if (posterLargeEl) {
      if (movie.poster_path) {
        const url = `https://image.tmdb.org/t/p/w500${movie.poster_path}`;
        posterLargeEl.innerHTML = `<img src="${url}" alt="${movie.title}">`;
      } else {
        posterLargeEl.innerHTML =
          '<div class="placeholder-poster large">Aucune affiche</div>';
      }
    }

    // 2) Films similaires (mode hybrid par défaut)
    if (similarContainer) {
      const simRes = await fetch(
        `${API_BASE}/recommend?title=${encodeURIComponent(
          movie.title || title
        )}&mode=hybrid&k=2`
      );
      if (simRes.ok) {
        const sims = await simRes.json();
        renderSimilar(similarContainer, movie.title || title, sims);
      } else {
        similarContainer.innerHTML =
          "<p class='empty'>Aucune recommandation disponible.</p>";
      }
    }
  } catch (err) {
    console.error("movie page error", err);
  }
}

function renderSimilar(container, sourceTitle, movies) {
  if (!movies || !movies.length) {
    container.innerHTML =
      "<p class='empty'>Aucune recommandation disponible.</p>";
    return;
  }

  const list = document.createElement("div");
  list.className = "similar-grid";

  movies.forEach((m) => {
    const card = document.createElement("article");
    card.className = "similar-card";

    const posterUrl = m.poster_path
      ? `https://image.tmdb.org/t/p/w300${m.poster_path}`
      : null;

    card.innerHTML = `
      <div class="similar-poster">
        ${
          posterUrl
            ? `<img src="${posterUrl}" alt="${m.title}">`
            : `<div class="placeholder-poster small">No poster</div>`
        }
      </div>
      <div class="similar-info">
        <h4>${m.title}</h4>
        <p class="similar-meta">${m.genres || ""}</p>
        <p class="similar-overview">${
          m.overview ? m.overview.slice(0, 140) + "…" : ""
        }</p>
        <button class="btn-link"
          onclick="window.location.href='movie.html?title=${encodeURIComponent(
            m.title
          )}'">
          Voir la fiche
        </button>
      </div>
    `;
    list.appendChild(card);
  });

  container.innerHTML = "";
  container.appendChild(list);
}

document.addEventListener("DOMContentLoaded", loadMoviePage);
