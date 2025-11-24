from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util as st_util
import os
from pathlib import Path

from vector_index import VectorIndex


CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class ContentModels:
    df: pd.DataFrame
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: any
    tfidf_knn: NearestNeighbors
    bert_model: SentenceTransformer
    bert_embeddings: np.ndarray
    bert_index: VectorIndex


def build_content_models(df: pd.DataFrame) -> ContentModels:
    # ---------- TF-IDF ----------
    tfidf = TfidfVectorizer(stop_words="english", max_features=50_000)
    tfidf_matrix = tfidf.fit_transform(df["text"].values)

    tfidf_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    tfidf_knn.fit(tfidf_matrix)

    # ---------- BERT + CACHE ----------
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    bert_model = SentenceTransformer(model_name)

    cache_path = CACHE_DIR / "bert_embeddings.npy"

    if cache_path.exists():
        print(">> Loading cached BERT embeddings…")
        bert_embeddings = np.load(cache_path)
        # sécurité : vérifier la taille correspond
        if bert_embeddings.shape[0] != len(df):
            print("!! Cache size mismatch, recomputing embeddings")
            bert_embeddings = rebuild_bert_cache(bert_model, df, cache_path)
    else:
        bert_embeddings = rebuild_bert_cache(bert_model, df, cache_path)

    bert_index = VectorIndex(bert_embeddings, metric="cosine")

    return ContentModels(
        df=df,
        tfidf_vectorizer=tfidf,
        tfidf_matrix=tfidf_matrix,
        tfidf_knn=tfidf_knn,
        bert_model=bert_model,
        bert_embeddings=bert_embeddings,
        bert_index=bert_index,
    )


def rebuild_bert_cache(bert_model, df, cache_path):
    print(">> Computing BERT embeddings (first time)…")
    embeddings = bert_model.encode(
        df["text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    print(">> Saving embeddings cache:", cache_path)
    np.save(cache_path, embeddings)
    return embeddings


# ------------------------------------------------------------
# Same recommend_tfidf / recommend_bert / explain_pair as before
# (garde exactement le même code)
# ------------------------------------------------------------

def find_movie_index(df: pd.DataFrame, title: str) -> Optional[int]:
    title = title.strip().lower()
    matches = df[df["title"].str.lower() == title]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title)]
        if matches.empty:
            return None
    return int(matches.index[0])


def recommend_tfidf(models: ContentModels, title: str, k: int = 10):
    idx = find_movie_index(models.df, title)
    if idx is None:
        raise ValueError("Title not found")

    vec = models.tfidf_matrix[idx]
    distances, indices = models.tfidf_knn.kneighbors(vec, n_neighbors=k + 1)
    distances, indices = distances[0], indices[0]

    out_idx = []
    out_sims = []
    for d, i in zip(distances, indices):
        if i == idx:
            continue
        out_idx.append(int(i))
        out_sims.append(1.0 - float(d))
        if len(out_idx) >= k:
            break

    return out_idx, np.array(out_sims, dtype="float32")


def recommend_bert(models: ContentModels, title: str, k: int = 10):
    idx = find_movie_index(models.df, title)
    if idx is None:
        raise ValueError("Title not found")

    vec = models.bert_embeddings[idx]
    indices, sims = models.bert_index.search(vec, k + 1)

    out_idx, out_sims = [], []
    for i, s in zip(indices, sims):
        if i == idx:
            continue
        out_idx.append(int(i))
        out_sims.append(float(s))
        if len(out_idx) >= k:
            break
    return out_idx, np.array(out_sims, dtype="float32")


def hybrid_scores(tfidf_sims, bert_sims, alpha=0.5):
    def norm(x):
        if x.size == 0:
            return x
        mi, ma = float(x.min()), float(x.max())
        if ma - mi < 1e-8:
            return np.ones_like(x) * 0.5
        return (x - mi) / (ma - mi)

    return alpha * norm(tfidf_sims) + (1 - alpha) * norm(bert_sims)

def explain_pair(models, idx_source: int, idx_cand: int) -> dict:
    df = models.df
    m_src = df.iloc[idx_source]
    m_cand = df.iloc[idx_cand]

    # --- TF-IDF similarity (cosine) ---
    v1 = models.tfidf_matrix[idx_source]
    v2 = models.tfidf_matrix[idx_cand]
    num = float((v1 @ v2.T).toarray()[0, 0])
    den = (
        float(np.linalg.norm(v1.toarray())) * float(np.linalg.norm(v2.toarray()))
        + 1e-9
    )
    tfidf_sim = num / den

    # --- BERT similarity ---
    bert_sim = float(
        st_util.cos_sim(
            models.bert_embeddings[idx_source],
            models.bert_embeddings[idx_cand],
        )[0][0]
    )

    # --- Fallback colonnes *_clean / brutes ---
    def _split_field(val):
        if isinstance(val, str):
            return [x.strip() for x in val.split(",") if x.strip()]
        return []

    genres_src = m_src.get("genres_clean") or m_src.get("genres") or ""
    genres_cand = m_cand.get("genres_clean") or m_cand.get("genres") or ""
    shared_genres = list(
        set(_split_field(genres_src)) & set(_split_field(genres_cand))
    )

    cast_src = m_src.get("cast_clean") or m_src.get("cast") or ""
    cast_cand = m_cand.get("cast_clean") or m_cand.get("cast") or ""
    shared_cast = list(
        set(_split_field(cast_src)) & set(_split_field(cast_cand))
    )

    director_src = m_src.get("director_clean") or m_src.get("director")
    director_cand = m_cand.get("director_clean") or m_cand.get("director")

    same_director = (
        isinstance(director_src, str)
        and isinstance(director_cand, str)
        and director_src == director_cand
    )

    return {
        "source_title": m_src["title"],
        "candidate_title": m_cand["title"],
        "similarity_tfidf": tfidf_sim,
        "similarity_bert": bert_sim,
        "shared_genres": shared_genres,
        "shared_cast": shared_cast[:5],
        "same_director": same_director,
        "director": director_src if isinstance(director_src, str) else None,
    }
