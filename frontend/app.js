const API_BASE = "http://127.0.0.1:8000";

const movieInput = document.getElementById("movie-input");
const suggestionsEl = document.getElementById("suggestions");
const resultsEl = document.getElementById("results");
const explanationEl = document.getElementById("explanation");
const searchBtn = document.getElementById("search-btn");

const accountBtn = document.getElementById("account-btn");
const accountMenu = document.getElementById("account-menu");
const authModal = document.getElementById("auth-modal");
const authModalClose = document.getElementById("auth-modal-close");

let debounceTimer = null;

// ---------- AUTOCOMPLETE ----------

movieInput.addEventListener("input", () => {
  const q = movieInput.value.trim();
  if (!q) {
    suggestionsEl.innerHTML = "";
    return;
  }
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => fetchSuggestions(q), 200);
});

movieInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    getRecommendations();
  }
});

suggestionsEl.addEventListener("click", (e) => {
  if (e.target && e.target.matches("li.suggestion-item")) {
    const title = e.target.dataset.title;
    movieInput.value = title;
    suggestionsEl.innerHTML = "";
    getRecommendations();
  }
});

searchBtn.addEventListener("click", () => {
  getRecommendations();
});

async function fetchSuggestions(query) {
  try {
    const res = await fetch(
      `${API_BASE}/suggest?query=${encodeURIComponent(query)}`
    );
    if (!res.ok) return;
    const data = await res.json();
    renderSuggestions(data);
  } catch (err) {
    console.error("suggest error", err);
  }
}

function renderSuggestions(list) {
  suggestionsEl.innerHTML = "";
  list.forEach((title) => {
    const li = document.createElement("li");
    li.className = "suggestion-item";
    li.textContent = title;
    li.dataset.title = title;
    suggestionsEl.appendChild(li);
  });
}

// ---------- RECOMMENDATIONS ----------

function getSelectedMode() {
  const el = document.querySelector('input[name="mode"]:checked');
  return el ? el.value : "tfidf";
}

async function getRecommendations() {
  const title = movieInput.value.trim();
  if (!title) return;

  explanationEl.innerHTML = "";
  resultsEl.innerHTML = `<div class="loading">Searching recommendations...</div>`;

  const mode = getSelectedMode();

  try {
    const res = await fetch(
      `${API_BASE}/recommend?title=${encodeURIComponent(
        title
      )}&mode=${mode}&k=10`
    );
    if (!res.ok) {
      resultsEl.innerHTML = `<div class="error">No results. (${res.status})</div>`;
      return;
    }
    const data = await res.json();
    renderResults(title, data, mode);
  } catch (err) {
    console.error("recommend error", err);
    resultsEl.innerHTML = `<div class="error">Error while fetching recommendations.</div>`;
  }
}

function renderResults(sourceTitle, movies, mode) {
  resultsEl.innerHTML = "";
  if (!movies.length) {
    if (mode === "cf") {
      resultsEl.innerHTML = `
        <div class="empty">
          Aucune recommandation collaborative disponible pour ce film.<br/>
          (Pas de données MovieLens 100K pour ce titre.)
        </div>`;
    } else {
      resultsEl.innerHTML = `<div class="empty">No recommendations found.</div>`;
    }
    return;
  }

  movies.forEach((m) => {
    const card = document.createElement("article");
    card.className = "movie-card";

    const posterUrl = m.poster_path
      ? `https://image.tmdb.org/t/p/w500${m.poster_path}`
      : null;

    card.innerHTML = `
      <div class="movie-poster">
        ${
          posterUrl
            ? `<img src="${posterUrl}" alt="${m.title}" />`
            : `<div class="placeholder-poster">No poster</div>`
        }
      </div>
      <div class="movie-info">
        <div class="movie-header-row">
          <div>
            <h3>
              <button type="button" class="movie-title-btn">${m.title}</button>
            </h3>
            <p class="movie-meta">
              ${
                m.genres
                  ? `<span class="badge">${m.genres}</span>`
                  : `<span class="badge badge-muted">No genres</span>`
              }
            </p>
          </div>
          <div class="card-actions">
            <button class="icon-btn watchlist-btn" title="Ajouter à ma watchlist">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M2 5H14V7H2V5Z" fill="currentColor" />
                <path d="M2 9H14V11H2V9Z" fill="currentColor" />
                <path d="M10 13H2V15H10V13Z" fill="currentColor" />
                <path d="M16 9H18V13H22V15H18V19H16V15H12V13H16V9Z" fill="currentColor" />
              </svg>
            </button>
            <div class="rating-btn-wrapper">
              <button class="icon-btn rating-btn" title="Noter ce film">
                <span class="icon-rating"></span>
              </button>
              <div class="rating-bubbles">
                <span data-rating="dislike"><span class="emoji emoji-dislike">👎</span> Pas aimé</span>
                <span data-rating="like"><span class="emoji emoji-like">👍</span> J'ai aimé</span>
                <span data-rating="love"><span class="emoji emoji-star-eyes">🤩</span> Adoré</span>
              </div>
            </div>
            <button class="icon-btn seen-toggle" data-state="unseen">
              <span class="seen-dot"></span>
              <span class="seen-label">Pas vu</span>
            </button>
          </div>
        </div>

        <p class="movie-overview">${
          m.overview || "No overview available."
        }</p>
        <div class="scores">
          ${formatScores(m)}
        </div>
        <div class="card-footer-row">
          <button class="btn-explain" data-source="${sourceTitle}" data-candidate="${m.title}">
            Explain
          </button>
        </div>
      </div>
    `;

    const btnExplain = card.querySelector(".btn-explain");
    btnExplain.addEventListener("click", () => {
      explainChoice(btnExplain.dataset.source, btnExplain.dataset.candidate);
    });

    const titleBtn = card.querySelector(".movie-title-btn");
    const detailsBtn = card.querySelector(".details-btn");
    const open = () => openMoviePage(m, sourceTitle);

    if (titleBtn) titleBtn.addEventListener("click", open);
    if (detailsBtn) detailsBtn.addEventListener("click", open);

    setupCardActions(card);
    resultsEl.appendChild(card);
  });
}

function formatScores(m) {
  const parts = [];
  if (m.score_tfidf != null) {
    parts.push(`TF-IDF: ${(m.score_tfidf * 100).toFixed(1)}%`);
  }
  if (m.score_bert != null) {
    parts.push(`BERT: ${(m.score_bert * 100).toFixed(1)}%`);
  }
  if (m.score_cf != null) {
    parts.push(`CF: ${(m.score_cf * 100).toFixed(1)}%`);
  }
  if (m.score_hybrid != null) {
    parts.push(`Hybrid: ${(m.score_hybrid * 100).toFixed(1)}%`);
  }
  if (!parts.length)
    return `<span class="score-chip score-muted">No scores</span>`;
  return parts.map((t) => `<span class="score-chip">${t}</span>`).join("");
}

// ---------- EXPLAIN ----------

async function explainChoice(source, candidate) {
  explanationEl.innerHTML = `<div class="loading">Explaining why "${candidate}" is recommended...</div>`;
  try {
    const res = await fetch(
      `${API_BASE}/explain?source=${encodeURIComponent(
        source
      )}&candidate=${encodeURIComponent(candidate)}`
    );
    if (!res.ok) {
      explanationEl.innerHTML = `<div class="error">Unable to explain this recommendation.</div>`;
      return;
    }
    const exp = await res.json();
    renderExplanation(exp);
  } catch (err) {
    console.error("explain error", err);
    explanationEl.innerHTML = `<div class="error">Error while fetching explanation.</div>`;
  }
}

function renderExplanation(exp) {
  explanationEl.innerHTML = `
    <div class="explanation-card">
      <h2>Why <span>${exp.candidate_title}</span> is recommended for <span>${exp.source_title}</span></h2>
      <p class="sim-metrics">
        <span>TF-IDF similarity: ${(exp.similarity_tfidf * 100).toFixed(
          1
        )}%</span>
        <span>BERT similarity: ${(exp.similarity_bert * 100).toFixed(1)}%</span>
      </p>
      <ul class="exp-list">
        <li>
          <strong>Shared genres:</strong>
          ${
            exp.shared_genres && exp.shared_genres.length
              ? exp.shared_genres.join(", ")
              : "None"
          }
        </li>
        <li>
          <strong>Shared cast (top):</strong>
          ${
            exp.shared_cast && exp.shared_cast.length
              ? exp.shared_cast.join(", ")
              : "None"
          }
        </li>
        <li>
          <strong>Director:</strong>
          ${
            exp.same_director
              ? `Same director: ${exp.director || "Unknown"}`
              : exp.director
              ? `Different directors (source: ${exp.director})`
              : "Unknown"
          }
        </li>
      </ul>
    </div>
  `;
}

// ---------- CARD ACTIONS (UI-only) ----------

function setupCardActions(card) {
  const watchlistBtn = card.querySelector(".watchlist-btn");
  if (watchlistBtn) {
    watchlistBtn.addEventListener("click", () => {
      watchlistBtn.classList.toggle("active");
    });
  }

  const ratingWrapper = card.querySelector(".rating-btn-wrapper");
  if (ratingWrapper) {
    const btn = ratingWrapper.querySelector(".rating-btn");
    const bubbles = ratingWrapper.querySelector(".rating-bubbles");
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      bubbles.classList.toggle("open");
    });
    bubbles.querySelectorAll("span").forEach((b) => {
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        const rating = b.dataset.rating;
        btn.dataset.rating = rating;
        bubbles.classList.remove("open");
      });
    });
    document.addEventListener("click", (e) => {
      if (!ratingWrapper.contains(e.target)) {
        bubbles.classList.remove("open");
      }
    });
  }

  const seenToggle = card.querySelector(".seen-toggle");
  if (seenToggle) {
    const label = seenToggle.querySelector(".seen-label");
    seenToggle.addEventListener("click", () => {
      const state = seenToggle.dataset.state === "seen" ? "unseen" : "seen";
      seenToggle.dataset.state = state;
      if (state === "seen") {
        label.textContent = "Vu";
        seenToggle.classList.add("seen-on");
      } else {
        label.textContent = "Pas vu";
        seenToggle.classList.remove("seen-on");
      }
    });
  }
}

// ---------- MOVIE PAGE (sessionStorage) ----------

function openMoviePage(movie, sourceTitle) {
  try {
    sessionStorage.setItem("cinematch_movie", JSON.stringify(movie));
    sessionStorage.setItem("cinematch_movie_source", sourceTitle || "");
  } catch (e) {
    console.warn("Unable to save movie in sessionStorage", e);
  }
  window.location.href = "movie.html";
}

// ---------- ACCOUNT BUBBLE / MODALE AUTH ----------

if (accountBtn && accountMenu) {
  accountBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    accountMenu.classList.toggle("open");
  });

  document.addEventListener("click", (e) => {
    if (!accountMenu.contains(e.target) && e.target !== accountBtn) {
      accountMenu.classList.remove("open");
    }
  });

  accountMenu.querySelectorAll(".menu-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.dataset.action;
      if (action === "profile") {
        window.location.href = "account.html";
      } else {
        openAuthModal(action);
      }
      accountMenu.classList.remove("open");
    });
  });
}

function openAuthModal(action) {
  const titleEl = document.getElementById("auth-modal-title");
  const subtitleEl = document.getElementById("auth-modal-subtitle");
  const bodyEl = document.getElementById("auth-modal-body");

  if (action === "login") {
    titleEl.textContent = "Let me connect !";
  } else if (action === "signup") {
    titleEl.textContent = "Log-in / Sign-up";
  } else {
    titleEl.textContent = "My account";
  }

  subtitleEl.textContent =
    "Cette interface est une maquette (pas de backend utilisateur pour l’instant).";
  bodyEl.innerHTML = renderAuthForm();

  authModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function renderAuthForm() {
  return `
    <form class="modal-form">
      <label>
        Email
        <input type="email" placeholder="you@example.com" />
      </label>
      <label>
        Mot de passe
        <input type="password" placeholder="••••••••" />
      </label>
      <button type="button" class="modal-primary-btn">Continuer</button>
    </form>
  `;
}

if (authModalClose) {
  authModalClose.addEventListener("click", () => {
    authModal.classList.add("hidden");
    document.body.classList.remove("modal-open");
  });
}

if (authModal) {
  authModal.addEventListener("click", (e) => {
    if (e.target.classList.contains("modal-backdrop")) {
      authModal.classList.add("hidden");
      document.body.classList.remove("modal-open");
    }
  });
}
