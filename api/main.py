from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

df = pd.read_csv("../data/data.csv")

# même filtrage que dans ton notebook
df = df[df["original_language"] == "en"]
df = df[df["overview"].notna()]
df = df.sort_values("popularity", ascending=False).head(30000).reset_index(drop=True)


def clean_text(col):
    if isinstance(col, str):
        return col.replace("[", "").replace("]", "").replace("'", "")
    return ""


df["genres_clean"] = df["genres"].apply(clean_text)
df["cast_clean"] = df["cast"].apply(clean_text)
df["director_clean"] = df["director"].fillna("")
df["tagline_clean"] = df["tagline"].fillna("")
df["overview_clean"] = df["overview"].fillna("")

df["text"] = (
    df["overview_clean"]
    + " "
    + df["genres_clean"]
    + " "
    + df["cast_clean"]
    + " "
    + df["director_clean"]
    + " "
    + df["tagline_clean"]
)

tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
tfidf_matrix = tfidf.fit_transform(df["text"])

knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(tfidf_matrix)


@app.get("/recommend")
def recommend(title: str, n: int = 10):
    title_low = title.lower()

    if title_low not in df["title"].str.lower().values:
        return {"error": f"Movie '{title}' not found."}

    idx = df[df["title"].str.lower() == title_low].index[0]
    vector = tfidf_matrix[idx]

    distances, indices = knn.kneighbors(vector, n_neighbors=n + 1)
    rec_df = df.iloc[indices[0][1:]][
        ["title", "overview", "genres_clean", "poster_path"]
    ]

    # IMPORTANT : enlever les NaN avant de convertir en dict
    rec_df = rec_df.where(pd.notnull(rec_df), None)

    return rec_df.to_dict(orient="records")


@app.get("/explain")
def explain(source: str, candidate: str):
    source_low = source.lower()
    cand_low = candidate.lower()

    if source_low not in df["title"].str.lower().values:
        return {"error": f"Source movie '{source}' not found."}
    if cand_low not in df["title"].str.lower().values:
        return {"error": f"Candidate movie '{candidate}' not found."}

    idx_s = df[df["title"].str.lower() == source_low].index[0]
    idx_c = df[df["title"].str.lower() == cand_low].index[0]

    # Similarité TF-IDF entre les deux films
    sim = float(cosine_similarity(tfidf_matrix[idx_s], tfidf_matrix[idx_c])[0, 0])

    # Genres en commun
    def to_set(val: str):
        if not isinstance(val, str):
            return set()
        return {g.strip() for g in val.split(",") if g.strip()}

    genres_s = to_set(df.loc[idx_s, "genres_clean"])
    genres_c = to_set(df.loc[idx_c, "genres_clean"])
    shared_genres = sorted(list(genres_s & genres_c))

    # Réalisateur
    dir_s = (df.loc[idx_s, "director_clean"] or "").strip()
    dir_c = (df.loc[idx_c, "director_clean"] or "").strip()
    same_director = bool(dir_s and dir_c and dir_s == dir_c)

    # Cast commun (on coupe aux 5 premiers pour éviter le bruit)
    cast_s = to_set(df.loc[idx_s, "cast_clean"])
    cast_c = to_set(df.loc[idx_c, "cast_clean"])
    shared_cast = sorted(list(cast_s & cast_c))[:5]

    return {
        "source": df.loc[idx_s, "title"],
        "candidate": df.loc[idx_c, "title"],
        "similarity": sim,
        "shared_genres": shared_genres,
        "same_director": same_director,
        "director": dir_c if same_director else dir_c,
        "shared_cast": shared_cast,
    }


@app.get("/suggest")
def suggest(query: str, limit: int = 10):
    q = query.strip().lower()
    if not q:
        return []

    mask = df["title"].str.lower().str.contains(q, na=False)
    suggestions = df.loc[mask, "title"].dropna().drop_duplicates().head(limit)

    return suggestions.to_list()
