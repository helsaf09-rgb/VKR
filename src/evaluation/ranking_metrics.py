from __future__ import annotations

import math

import pandas as pd


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(len(relevant))


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")

    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / float(i)

    if hits == 0:
        return 0.0
    return score / float(min(len(relevant), k))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")

    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / math.log2(i + 1.0)

    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_len + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_ranking(
    recommendations_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    k: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if recommendations_df.empty:
        metrics = {
            "k": int(k),
            "n_users_evaluated": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }
        empty_per_user = pd.DataFrame(
            columns=["user_id", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]
        )
        return metrics, empty_per_user

    gt_positive = ground_truth_df[["user_id", "offer_id"]].drop_duplicates()
    gt_by_user = gt_positive.groupby("user_id")["offer_id"].apply(set).to_dict()

    rec_sorted = recommendations_df.sort_values(["user_id", "rank"])
    rec_by_user = rec_sorted.groupby("user_id")["offer_id"].apply(list).to_dict()

    user_ids = sorted(set(gt_by_user).intersection(set(rec_by_user)))
    rows: list[dict[str, float | str]] = []
    for user_id in user_ids:
        recommended = rec_by_user[user_id]
        relevant = gt_by_user[user_id]
        rows.append(
            {
                "user_id": user_id,
                "precision_at_k": precision_at_k(recommended, relevant, k),
                "recall_at_k": recall_at_k(recommended, relevant, k),
                "map_at_k": average_precision_at_k(recommended, relevant, k),
                "ndcg_at_k": ndcg_at_k(recommended, relevant, k),
            }
        )

    per_user_df = pd.DataFrame(rows)
    if per_user_df.empty:
        metrics = {
            "k": int(k),
            "n_users_evaluated": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }
        return metrics, per_user_df

    metrics = {
        "k": int(k),
        "n_users_evaluated": int(len(per_user_df)),
        "precision_at_k": float(per_user_df["precision_at_k"].mean()),
        "recall_at_k": float(per_user_df["recall_at_k"].mean()),
        "map_at_k": float(per_user_df["map_at_k"].mean()),
        "ndcg_at_k": float(per_user_df["ndcg_at_k"].mean()),
    }
    return metrics, per_user_df
