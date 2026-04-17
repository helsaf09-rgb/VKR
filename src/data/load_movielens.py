from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens_100k(download_dir: Path, force_download: bool = False) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "ml-100k.zip"
    extracted_dir = download_dir / "ml-100k"
    marker = extracted_dir / "u.data"

    if marker.exists() and not force_download:
        return extracted_dir

    if force_download or not zip_path.exists():
        urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(download_dir)

    if not marker.exists():
        raise FileNotFoundError(f"MovieLens extraction failed: {marker} not found.")
    return extracted_dir


def load_movielens_implicit(
    dataset_dir: Path,
    min_rating: int = 4,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ratings = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
    )

    filtered = ratings[ratings["rating"] >= min_rating].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], unit="s")

    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_id").size()
        filtered = filtered[filtered["user_id"].map(user_counts) >= min_user_interactions].copy()
        item_counts = filtered.groupby("item_id").size()
        filtered = filtered[filtered["item_id"].map(item_counts) >= min_item_interactions].copy()
        if len(filtered) == before:
            break

    interactions = filtered.rename(columns={"item_id": "offer_id"}).copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["offer_id"] = interactions["offer_id"].astype(str)
    interactions["label"] = 1

    offers = items.rename(columns={"item_id": "offer_id", "title": "offer_name"}).copy()
    offers["offer_id"] = offers["offer_id"].astype(str)
    offers = offers[offers["offer_id"].isin(interactions["offer_id"].unique())].reset_index(drop=True)

    stats: dict[str, Any] = {
        "dataset_name": "MovieLens 100K",
        "source_url": MOVIELENS_100K_URL,
        "min_rating": int(min_rating),
        "min_user_interactions": int(min_user_interactions),
        "min_item_interactions": int(min_item_interactions),
        "n_users": int(interactions["user_id"].nunique()),
        "n_items": int(interactions["offer_id"].nunique()),
        "n_positive_interactions": int(len(interactions)),
        "timestamp_min": interactions["timestamp"].min().isoformat(),
        "timestamp_max": interactions["timestamp"].max().isoformat(),
    }
    return interactions, offers, stats
