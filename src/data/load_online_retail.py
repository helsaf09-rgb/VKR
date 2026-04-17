from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ONLINE_RETAIL_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
ONLINE_RETAIL_KAGGLE_URL = "https://www.kaggle.com/datasets/ineubytes/online-retail-ecommerce-dataset"


def download_online_retail_xlsx(download_dir: Path, force_download: bool = False) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = download_dir / "Online Retail.xlsx"

    if dataset_path.exists() and not force_download:
        return dataset_path

    urllib.request.urlretrieve(ONLINE_RETAIL_UCI_URL, dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Online Retail download failed: {dataset_path} not found.")
    return dataset_path


def _iterative_interaction_filter(
    interactions_df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    max_users: int | None,
    max_items: int | None,
) -> pd.DataFrame:
    filtered = interactions_df.copy()

    while True:
        before = (
            int(len(filtered)),
            int(filtered["user_id"].nunique()),
            int(filtered["offer_id"].nunique()),
        )

        user_counts = filtered.groupby("user_id").size()
        filtered = filtered[filtered["user_id"].map(user_counts) >= min_user_interactions].copy()

        item_counts = filtered.groupby("offer_id").size()
        filtered = filtered[filtered["offer_id"].map(item_counts) >= min_item_interactions].copy()

        if max_items is not None and filtered["offer_id"].nunique() > max_items:
            keep_items = (
                filtered.groupby("offer_id")["amount"]
                .sum()
                .sort_values(ascending=False)
                .head(max_items)
                .index
            )
            filtered = filtered[filtered["offer_id"].isin(keep_items)].copy()

        if max_users is not None and filtered["user_id"].nunique() > max_users:
            keep_users = (
                filtered.groupby("user_id")["amount"]
                .sum()
                .sort_values(ascending=False)
                .head(max_users)
                .index
            )
            filtered = filtered[filtered["user_id"].isin(keep_users)].copy()

        after = (
            int(len(filtered)),
            int(filtered["user_id"].nunique()),
            int(filtered["offer_id"].nunique()),
        )
        if after == before:
            return filtered


def load_online_retail_implicit(
    dataset_path: Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 10,
    min_purchase_value: float = 0.0,
    max_users: int | None = 3000,
    max_items: int | None = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_df = pd.read_excel(dataset_path, engine="openpyxl")
    raw_rows = len(raw_df)

    tx = raw_df.rename(
        columns={
            "CustomerID": "customer_id",
            "StockCode": "stock_code",
            "Description": "description",
            "InvoiceDate": "invoice_date",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
            "Country": "country",
        }
    ).copy()

    tx = tx.dropna(subset=["customer_id", "stock_code", "description", "invoice_date"]).copy()
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx["unit_price"] = pd.to_numeric(tx["unit_price"], errors="coerce")
    tx = tx[(tx["quantity"] > 0) & (tx["unit_price"] > 0)].copy()
    tx["timestamp"] = pd.to_datetime(tx["invoice_date"], errors="coerce")
    tx = tx.dropna(subset=["timestamp"]).copy()

    tx["user_id"] = tx["customer_id"].astype(int).astype(str)
    tx["offer_id"] = tx["stock_code"].astype(str).str.strip()
    tx["offer_name"] = tx["description"].astype(str).str.strip()
    tx["amount"] = tx["quantity"] * tx["unit_price"]
    tx = tx[tx["amount"] >= float(min_purchase_value)].copy()

    cleaned_rows = len(tx)
    unique_pairs_before = int(tx[["user_id", "offer_id"]].drop_duplicates().shape[0])

    aggregated = (
        tx.sort_values("timestamp")
        .groupby(["user_id", "offer_id"], as_index=False)
        .agg(
            timestamp=("timestamp", "max"),
            amount=("amount", "sum"),
            quantity=("quantity", "sum"),
            offer_name=("offer_name", "last"),
            country=("country", "last"),
        )
    )

    filtered = _iterative_interaction_filter(
        interactions_df=aggregated,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        max_users=max_users,
        max_items=max_items,
    )

    interactions = filtered[["user_id", "offer_id", "timestamp", "amount", "quantity"]].copy()
    interactions["label"] = 1
    interactions = interactions.sort_values(["user_id", "timestamp", "offer_id"]).reset_index(drop=True)

    offers = (
        filtered.sort_values("timestamp")
        .groupby("offer_id", as_index=False)
        .agg(
            offer_name=("offer_name", "last"),
            total_amount=("amount", "sum"),
            total_quantity=("quantity", "sum"),
            buyer_count=("user_id", "nunique"),
        )
        .sort_values(["buyer_count", "total_amount"], ascending=[False, False])
        .reset_index(drop=True)
    )

    stats: dict[str, Any] = {
        "dataset_name": "Online Retail",
        "dataset_origin": "UCI Machine Learning Repository",
        "source_url": ONLINE_RETAIL_UCI_URL,
        "kaggle_url": ONLINE_RETAIL_KAGGLE_URL,
        "min_user_interactions": int(min_user_interactions),
        "min_item_interactions": int(min_item_interactions),
        "min_purchase_value": float(min_purchase_value),
        "max_users": None if max_users is None else int(max_users),
        "max_items": None if max_items is None else int(max_items),
        "raw_rows": int(raw_rows),
        "cleaned_rows": int(cleaned_rows),
        "unique_user_item_pairs_before_filtering": int(unique_pairs_before),
        "n_users": int(interactions["user_id"].nunique()),
        "n_items": int(interactions["offer_id"].nunique()),
        "n_positive_interactions": int(len(interactions)),
        "timestamp_min": interactions["timestamp"].min().isoformat(),
        "timestamp_max": interactions["timestamp"].max().isoformat(),
        "total_amount_sum": float(interactions["amount"].sum()),
    }
    return interactions, offers, stats
