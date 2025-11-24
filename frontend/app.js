let currentQueryTitle = null;
let suggestTimeout = null;

function onTitleInput(e) {
    const value = e.target.value.trim();

    // debounce pour éviter de spammer l'API
    clearTimeout(suggestTimeout);
    if (!value) {
        document.getElementById("suggestions").innerHTML = "";
        return;
    }

    suggestTimeout = setTimeout(() => fetchSuggestions(value), 200);
}

async function fetchSuggestions(query) {
    const res = await fetch(`http://127.0.0.1:8000/suggest?query=${encodeURIComponent(query)}`);
    const data = await res.json();

    const sugDiv = document.getElementById("suggestions");
    sugDiv.innerHTML = "";

    data.forEach(title => {
        const div = document.createElement("div");
        div.classList.add("suggestion-item");
        div.textContent = title;
        div.onclick = () => {
            document.getElementById("movie-input").value = title;
            sugDiv.innerHTML = "";
            getRecommendations();
        };
        sugDiv.appendChild(div);
    });
}


async function getRecommendations() {
    const title = document.getElementById("movie-input").value.trim();
    const resultsDiv = document.getElementById("results");
    currentQueryTitle = title;

    if (!title) {
        resultsDiv.innerHTML = "<p>Please enter a movie title.</p>";
        return;
    }

    resultsDiv.innerHTML = "<p>Loading...</p>";

    const res = await fetch(`http://127.0.0.1:8000/recommend?title=${encodeURIComponent(title)}`);
    const data = await res.json();

    if (data.error) {
        resultsDiv.innerHTML = `<p class="error">${data.error}</p>`;
        return;
    }

    resultsDiv.innerHTML = "";

    data.forEach(item => {
        const div = document.createElement("div");
        div.classList.add("result-item");

        // URL de l'affiche TMDb
        let posterUrl = "";
        if (item.poster_path) {
            posterUrl = `https://image.tmdb.org/t/p/w300${item.poster_path}`;
        }

        const safeId = item.title.replace(/[^a-z0-9]/gi, "");

        div.innerHTML = `
            ${posterUrl ? `<img class="poster" src="${posterUrl}" alt="Poster of ${item.title}">` : ""}
            <div class="result-content">
                <h3>${item.title}</h3>
                <p><strong>Genres:</strong> ${item.genres_clean || "N/A"}</p>
                <p>${item.overview || ""}</p>
                <button class="explain-btn" onclick="explainChoice('${item.title}', '${safeId}')">Explain</button>
                <div id="explain-${safeId}" class="explanation"></div>
            </div>
        `;

        resultsDiv.appendChild(div);
    });
}

async function explainChoice(candidateTitle, safeId) {
    const container = document.getElementById("explain-" + safeId);
    container.innerHTML = "<p>Loading explanation...</p>";

    const res = await fetch(
        `http://127.0.0.1:8000/explain?source=${encodeURIComponent(currentQueryTitle)}&candidate=${encodeURIComponent(candidateTitle)}`
    );
    const data = await res.json();

    if (data.error) {
        container.innerHTML = `<p>${data.error}</p>`;
        return;
    }

    const simPercent = (data.similarity * 100).toFixed(1);

    let reasons = [];

    if (data.shared_genres && data.shared_genres.length > 0) {
        reasons.push(`They share the following genres: <strong>${data.shared_genres.join(", ")}</strong>.`);
    }

    if (data.same_director) {
        reasons.push(`They have the same director: <strong>${data.director}</strong>.`);
    }

    if (data.shared_cast && data.shared_cast.length > 0) {
        reasons.push(`They share some cast members: <strong>${data.shared_cast.join(", ")}</strong>.`);
    }

    if (reasons.length === 0) {
        reasons.push("This movie has a storyline and description very close to the one you entered, based on text similarity.");
    }

    container.innerHTML = `
        <p><strong>Why this movie?</strong></p>
        <ul>
            ${reasons.map(r => `<li>${r}</li>`).join("")}
        </ul>
        <p>Overall similarity score (TF-IDF cosine): <strong>${simPercent}%</strong></p>
    `;
}
