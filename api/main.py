from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import (
    build_movielens_to_tmdb_map,
    load_movielens_100k,
    load_tmdb_movies,
)
from models_content import (
    ContentModels,
    build_content_models,
    explain_pair,
    find_movie_index,
    hybrid_scores,
    recommend_bert,
    recommend_tfidf,
)
from models_cf import CFModels, build_cf_models, recommend_similar_items_cf

# ------------ chemins robustes ------------
BASE_DIR = Path(__file__).resolve().parent.parent
TMDB_PATH = BASE_DIR / "eda" / "tmdb_movies.csv"
MOVIELENS_DIR = BASE_DIR / "eda" / "movielens"

K_DEFAULT = 10
# -----------------------------------------


class MovieOut(BaseModel):
    title: str
    overview: Optional[str] = None
    genres: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    tagline: Optional[str] = None
    poster_path: Optional[str] = None
    score_tfidf: Optional[float] = None
    score_bert: Optional[float] = None
    score_cf: Optional[float] = None
    score_hybrid: Optional[float] = None


class ExplanationOut(BaseModel):
    source_title: str
    candidate_title: str
    similarity_tfidf: float
    similarity_bert: float
    shared_genres: List[str]
    shared_cast: List[str]
    same_director: bool
    director: Optional[str]


app = FastAPI(title="CineMatch v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre éventuellement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Chargement global des modèles ---------
TMDB_DF = load_tmdb_movies(TMDB_PATH)
CONTENT_MODELS: ContentModels = build_content_models(TMDB_DF)

ML_RATINGS, ML_ITEMS = load_movielens_100k(MOVIELENS_DIR)
ML_TO_TMDB = build_movielens_to_tmdb_map(ML_ITEMS, TMDB_DF)
CF_MODELS: CFModels = build_cf_models(ML_RATINGS, ML_ITEMS, ML_TO_TMDB)
# -------------------------------------------------


def movie_to_out(idx: int) -> MovieOut:
  row = TMDB_DF.iloc[idx]

  def clean(val):
      if val is None:
          return None
      try:
          import math
          if isinstance(val, float) and math.isnan(val):
              return None
      except Exception:
          pass
      return val

  # Fallback propre sur les colonnes
  genres = row.get("genres_clean")
  if not isinstance(genres, str) or not genres.strip():
      genres = row.get("genres")

  cast = row.get("cast_clean")
  if not isinstance(cast, str) or not cast.strip():
      cast = row.get("cast")

  director = row.get("director_clean")
  if not isinstance(director, str) or not director.strip():
      director = row.get("director")

  tagline = row.get("tagline_clean")
  if not isinstance(tagline, str) or not tagline.strip():
      tagline = row.get("tagline")

  return MovieOut(
      title=str(row["title"]),
      overview=clean(row.get("overview")),
      genres=clean(genres),
      director=clean(director),
      cast=clean(cast),
      tagline=clean(tagline),
      poster_path=clean(row.get("poster_path")),
  )

@app.get("/suggest", response_model=List[str])
def suggest(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    q = query.strip().lower()
    if not q:
        return []
    mask = TMDB_DF["title"].str.lower().str.contains(q, na=False)
    titles = (
        TMDB_DF.loc[mask, "title"]
        .dropna()
        .drop_duplicates()
        .head(limit)
        .tolist()
    )
    return titles


@app.get("/recommend", response_model=List[MovieOut])
def recommend(
    title: str = Query(..., min_length=1),
    mode: str = Query("tfidf", regex="^(tfidf|bert|cf|hybrid)$"),
    k: int = Query(K_DEFAULT, ge=1, le=50),
):
    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Empty title")

    # --- TF-IDF ---
    if mode == "tfidf":
        try:
            idxs, sims = recommend_tfidf(CONTENT_MODELS, title, k=k)
        except ValueError as e:
            # titre introuvable → 404 cohérent
            raise HTTPException(status_code=404, detail=str(e))
        movies = [movie_to_out(i) for i in idxs]
        for m, s in zip(movies, sims):
            m.score_tfidf = float(s)
        return movies

    # --- BERT ---
    if mode == "bert":
        try:
            idxs, sims = recommend_bert(CONTENT_MODELS, title, k=k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        movies = [movie_to_out(i) for i in idxs]
        for m, s in zip(movies, sims):
            m.score_bert = float(s)
        return movies

    # --- CF ---
    if mode == "cf":
        idxs = recommend_similar_items_cf(CF_MODELS, TMDB_DF, title, k=k)
        # si pas de mapping MovieLens → liste vide, pas d'erreur
        if not idxs:
            return []
        movies = [movie_to_out(i) for i in idxs]
        for m in movies:
            m.score_cf = 1.0
        return movies

    # --- HYBRID (TF-IDF + BERT) ---
    if mode == "hybrid":
        # Si TF-IDF ou BERT plantent pour ce titre, on laisse remonter en 404.
        try:
            idxs_tfidf, sims_tfidf = recommend_tfidf(CONTENT_MODELS, title, k=k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=f"TF-IDF: {e}")

        try:
            idxs_bert, sims_bert = recommend_bert(CONTENT_MODELS, title, k=k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=f"BERT: {e}")

        all_indices = sorted(set(idxs_tfidf) | set(idxs_bert))
        if not all_indices:
            return []

        tfidf_map = {i: s for i, s in zip(idxs_tfidf, sims_tfidf)}
        bert_map = {i: s for i, s in zip(idxs_bert, sims_bert)}

        tfidf_s = np.array([tfidf_map.get(i, 0.0) for i in all_indices])
        bert_s = np.array([bert_map.get(i, 0.0) for i in all_indices])

        hybrid_s = hybrid_scores(tfidf_s, bert_s, alpha=0.5)
        order = np.argsort(-hybrid_s)

        final_indices = [all_indices[i] for i in order[:k]]
        final_scores = hybrid_s[order[:k]]

        movies = [movie_to_out(i) for i in final_indices]
        for m, s in zip(movies, final_scores):
            m.score_hybrid = float(s)
        return movies

    # ne devrait pas arriver
    raise HTTPException(status_code=400, detail="Invalid mode")
    

@app.get("/explain", response_model=ExplanationOut)
def explain(
    source: str = Query(..., min_length=1),
    candidate: str = Query(..., min_length=1),
):
    idx_source = find_movie_index(TMDB_DF, source)
    idx_cand = find_movie_index(TMDB_DF, candidate)
    if idx_source is None or idx_cand is None:
        raise HTTPException(status_code=404, detail="Movie title not found")

    data = explain_pair(CONTENT_MODELS, idx_source, idx_cand)
    return ExplanationOut(**data)
