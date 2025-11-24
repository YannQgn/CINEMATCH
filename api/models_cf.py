from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class CFModels:
    ratings: pd.DataFrame
    items: pd.DataFrame
    item_user_matrix: csr_matrix
    item_knn: NearestNeighbors
    ml_to_tmdb: Dict[int, int]
    user_id_to_idx: Dict[int, int]
    item_id_to_idx: Dict[int, int]
    idx_to_item_id: Dict[int, int]


def build_cf_models(
    ratings: pd.DataFrame, items: pd.DataFrame, ml_to_tmdb: Dict[int, int]
) -> CFModels:
    """
    Item-based CF (cosine sur matrice item-user).
    """
    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())

    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_id_to_idx = {m: i for i, m in enumerate(item_ids)}

    rows = ratings["item_id"].map(item_id_to_idx).values
    cols = ratings["user_id"].map(user_id_to_idx).values
    data = ratings["rating"].values.astype("float32")

    n_items = len(item_ids)
    n_users = len(user_ids)

    item_user = csr_matrix((data, (rows, cols)), shape=(n_items, n_users))

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(item_user)

    idx_to_item_id = {i: m for m, i in item_id_to_idx.items()}

    return CFModels(
        ratings=ratings,
        items=items,
        item_user_matrix=item_user,
        item_knn=knn,
        ml_to_tmdb=ml_to_tmdb,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=idx_to_item_id,
    )


def recommend_similar_items_cf(
    cf: CFModels, tmdb_df: pd.DataFrame, title: str, k: int = 10
) -> List[int]:
    """
    Renvoie des indices TMDB recommandés par similarité CF (item-based).
    Si le film n'est pas mappé à MovieLens → renvoie simplement [].
    """
    title_norm = title.strip().lower()
    matches = tmdb_df[tmdb_df["title"].str.lower() == title_norm]
    if matches.empty:
        matches = tmdb_df[tmdb_df["title"].str.lower().str.contains(title_norm)]
        if matches.empty:
            # rien trouvé côté TMDB → pas d'erreur, juste aucune reco CF
            return []

    tmdb_idx = int(matches.index[0])

    # TMDB idx -> MovieLens item_id via mapping inverse
    inv_map = {v: k for k, v in cf.ml_to_tmdb.items()}
    ml_item_id = inv_map.get(tmdb_idx)
    if ml_item_id is None:
        # pas de correspondance dans MovieLens → pas de CF
        return []

    item_idx = cf.item_id_to_idx.get(ml_item_id)
    if item_idx is None:
        return []

    vec = cf.item_user_matrix[item_idx]
    distances, indices = cf.item_knn.kneighbors(vec, n_neighbors=k + 1)
    distances, indices = distances[0], indices[0]

    tmdb_indices: List[int] = []
    for d, i in zip(distances, indices):
        if i == item_idx:
            continue
        neighbor_item_id = cf.idx_to_item_id[i]
        tmdb_idx_neighbor = cf.ml_to_tmdb.get(neighbor_item_id)
        if tmdb_idx_neighbor is not None:
            tmdb_indices.append(int(tmdb_idx_neighbor))
        if len(tmdb_indices) >= k:
            break

    return tmdb_indices
