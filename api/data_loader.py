from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

MAX_MOVIES = 30_000  # limite pour éviter de tout charger (544k+ films)


def load_tmdb_movies(path: Path) -> pd.DataFrame:
    """
    Charge le CSV TMDB préparé (équivalent data.csv de ta V1).
    On attend au moins:
    - title
    - overview
    - original_language
    Éventuellement :
    - genres_clean, cast_clean, director_clean, tagline_clean, poster_path, release_date, popularity
    """
    df = pd.read_csv(path)

    # Filtrage de base
    if "original_language" in df.columns:
        df = df[df["original_language"] == "en"]
    if "overview" in df.columns:
        df = df[df["overview"].notnull()]

    # On garde seulement les films les plus populaires (sinon trop gros)
    if "popularity" in df.columns:
        df = (
            df.sort_values("popularity", ascending=False)
            .head(MAX_MOVIES)
            .reset_index(drop=True)
        )
    else:
        df = df.head(MAX_MOVIES).reset_index(drop=True)

    # Construction du champ texte global pour TF-IDF / BERT
    def col_or_none(name: str):
        return df[name] if name in df.columns else None

    # Fallback propre entre colonnes *_clean et colonnes brutes
    overview_col = col_or_none("overview_clean") or col_or_none("overview")
    genres_col = col_or_none("genres_clean") or col_or_none("genres")
    cast_col = col_or_none("cast_clean") or col_or_none("cast")
    director_col = col_or_none("director_clean") or col_or_none("director")
    tagline_col = col_or_none("tagline_clean") or col_or_none("tagline")

    text_parts = [overview_col, genres_col, cast_col, director_col, tagline_col]

    text_concat = []
    for i in range(len(df)):
        pieces = []
        for col in text_parts:
            if col is not None:
                val = col.iloc[i]
                if isinstance(val, str) and val.strip():
                    pieces.append(val.strip())
        text_concat.append(" ".join(pieces))

    df["text"] = text_concat
    return df


def load_movielens_100k(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge MovieLens 100k:
    - ratings: [user_id, item_id, rating]
    - items: [item_id, title, year]
    """
    ratings_path = base_dir / "u.data"
    items_path = base_dir / "u.item"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        usecols=["user_id", "item_id", "rating"],
    )

    items = pd.read_csv(
        items_path,
        sep="|",
        header=None,
        encoding="latin-1",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            *[f"genre_{i}" for i in range(19)],
        ],
        usecols=["item_id", "title", "release_date"],
    )

    def extract_year(date_str) -> Optional[int]:
        if isinstance(date_str, str) and len(date_str) >= 4:
            try:
                return int(date_str[-4:])
            except ValueError:
                return None
        return None

    items["year"] = items["release_date"].apply(extract_year)
    return ratings, items


def build_movielens_to_tmdb_map(
    ml_items: pd.DataFrame, tmdb_df: pd.DataFrame
) -> Dict[int, int]:
    """
    Map MovieLens item_id -> index TMDB (ligne dans tmdb_df) via (title, year) approx.
    Approche simple: titre normalisé, puis année si dispo.
    """
    tmdb = tmdb_df.copy()
    tmdb["title_norm"] = tmdb["title"].str.lower().str.strip()

    has_release = "release_date" in tmdb.columns
    mapping: Dict[int, int] = {}

    for _, row in ml_items.iterrows():
        item_id = int(row["item_id"])
        title_norm = str(row["title"]).lower().strip()
        year = row.get("year")

        candidates = tmdb[tmdb["title_norm"] == title_norm]

        if has_release and year and not candidates.empty:
            year_str = str(year)
            # ICI la correction : on filtre sur candidates, pas sur tmdb
            mask_year = candidates["release_date"].fillna("").astype(str).str.contains(year_str)
            candidates = candidates[mask_year]

        if not candidates.empty:
            idx = int(candidates.index[0])
            mapping[item_id] = idx

    return mapping