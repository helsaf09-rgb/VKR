from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.implicit_mf_recommender import ImplicitMFRecommender


def build_test_ground_truth(interactions_df: pd.DataFrame) -> pd.DataFrame:
    positives = interactions_df[interactions_df["label"] == 1].copy()
    if positives.empty:
        return pd.DataFrame(columns=["user_id", "offer_id", "timestamp"])

    positives["timestamp"] = pd.to_datetime(positives["timestamp"])
    test_gt = (
        positives.sort_values("timestamp")
        .groupby("user_id", as_index=False)
        .tail(1)[["user_id", "offer_id", "timestamp"]]
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )
    return test_gt


def build_exclusion_map(
    interactions_df: pd.DataFrame,
    test_ground_truth_df: pd.DataFrame,
) -> dict[str, set[str]]:
    positives = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].copy()
    holdout_pairs = set(
        tuple(x) for x in test_ground_truth_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )

    exclusion_map: dict[str, set[str]] = {}
    for user_id, offer_id in positives.itertuples(index=False):
        pair = (str(user_id), str(offer_id))
        if pair in holdout_pairs:
            continue
        exclusion_map.setdefault(str(user_id), set()).add(str(offer_id))
    return exclusion_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run implicit MF baseline on synthetic interactions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory containing interactions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for metrics and predictions.",
    )
    parser.add_argument("--n-factors", type=int, default=12, help="Latent factors for SVD.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SVD.")
    return parser.parse_args()


def build_train_interactions(
    interactions_df: pd.DataFrame,
    test_ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    holdout_pairs = set(
        tuple(x) for x in test_ground_truth_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )
    mask = interactions_df.apply(
        lambda r: (str(r["user_id"]), str(r["offer_id"])) not in holdout_pairs,
        axis=1,
    )
    return interactions_df.loc[mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = args.data_dir / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(
            f"{interactions_path} not found. Run run_baseline_pipeline first to generate synthetic data."
        )

    interactions_df = pd.read_csv(interactions_path)
    test_gt_df = build_test_ground_truth(interactions_df)
    train_interactions_df = build_train_interactions(interactions_df, test_gt_df)
    exclusion_map = build_exclusion_map(interactions_df, test_gt_df)
    user_ids = sorted(test_gt_df["user_id"].astype(str).unique().tolist())

    model = ImplicitMFRecommender(n_factors=args.n_factors, random_state=args.seed).fit(train_interactions_df)
    recommendations_df = model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )

    metrics, per_user_df = evaluate_ranking(
        recommendations_df=recommendations_df,
        ground_truth_df=test_gt_df[["user_id", "offer_id"]],
        k=args.top_k,
    )

    with (args.output_dir / "mf_baseline_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "mf_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "mf_per_user_metrics.csv", index=False)

    print("[ok] implicit MF baseline finished")
    print(f"[ok] evaluated users: {metrics['n_users_evaluated']}")
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
