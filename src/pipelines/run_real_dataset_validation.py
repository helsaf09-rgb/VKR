from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REAL_DATA_DIR, REPORTS_DIR
from src.data.load_movielens import download_movielens_100k, load_movielens_implicit
from src.data.load_online_retail import download_online_retail_xlsx, load_online_retail_implicit
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.implicit_mf_recommender import ImplicitMFRecommender
from src.models.item_knn_recommender import ImplicitItemKNNRecommender
from src.models.lightgcn_recommender import LightGCNConfig, LightGCNRecommender
from src.models.neural_cf_recommender import NeuralCFConfig, NeuralCFRecommender
from src.models.popularity_recommender import PopularityRecommender


def parse_hidden_dims(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError("At least one hidden layer size must be provided for Neural CF.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate recommender baselines on a real dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="online_retail",
        choices=["online_retail", "movielens"],
        help="Real dataset used for validation.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REAL_DATA_DIR / "online_retail",
        help="Directory where the raw dataset is downloaded and extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "real_validation",
        help="Directory where benchmark artifacts are stored.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations for evaluation.")
    parser.add_argument("--min-rating", type=int, default=4, help="Minimum explicit rating treated as positive.")
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Minimum number of positives required per user after filtering.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=10,
        help="Minimum number of positives required per item after filtering.",
    )
    parser.add_argument(
        "--min-purchase-value",
        type=float,
        default=0.0,
        help="Minimum positive transaction amount for the online retail dataset.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=3000,
        help="Optional upper bound on users retained after filtering for tractable validation.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1500,
        help="Optional upper bound on items retained after filtering for tractable validation.",
    )
    parser.add_argument("--n-factors", type=int, default=32, help="Latent factors for MF baseline.")
    parser.add_argument("--n-neighbors", type=int, default=50, help="Neighbors for item-kNN baseline.")
    parser.add_argument("--ncf-embedding-dim", type=int, default=16, help="Embedding dimension for Neural CF.")
    parser.add_argument(
        "--ncf-hidden-dims",
        type=str,
        default="32,16",
        help="Comma-separated hidden dimensions for Neural CF, e.g. '32,16'.",
    )
    parser.add_argument("--ncf-learning-rate", type=float, default=0.01, help="Learning rate for Neural CF.")
    parser.add_argument("--ncf-epochs", type=int, default=6, help="Epoch count for Neural CF.")
    parser.add_argument("--ncf-batch-size", type=int, default=1024, help="Batch size for Neural CF.")
    parser.add_argument(
        "--ncf-negative-samples",
        type=int,
        default=2,
        help="Number of sampled negatives per positive example in Neural CF.",
    )
    parser.add_argument("--lightgcn-embedding-dim", type=int, default=24, help="Embedding dimension for LightGCN.")
    parser.add_argument("--lightgcn-layers", type=int, default=2, help="Number of propagation layers in LightGCN.")
    parser.add_argument("--lightgcn-learning-rate", type=float, default=0.03, help="Learning rate for LightGCN.")
    parser.add_argument("--lightgcn-epochs", type=int, default=15, help="Epoch count for LightGCN.")
    parser.add_argument("--lightgcn-batch-size", type=int, default=4096, help="Batch size for LightGCN BPR.")
    parser.add_argument(
        "--lightgcn-samples-per-epoch",
        type=int,
        default=60000,
        help="Number of sampled BPR triples per epoch in LightGCN.",
    )
    parser.add_argument("--lightgcn-l2-reg", type=float, default=1e-4, help="L2 regularization for LightGCN.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if it already exists locally.",
    )
    return parser.parse_args()


def build_holdout(interactions_df: pd.DataFrame) -> pd.DataFrame:
    positives = interactions_df[interactions_df["label"] == 1].copy()
    positives["timestamp"] = pd.to_datetime(positives["timestamp"])
    deduplicated = (
        positives.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "offer_id"], keep="last")
        .sort_values("timestamp")
    )
    holdout = (
        deduplicated.groupby("user_id", as_index=False)
        .tail(1)[["user_id", "offer_id", "timestamp"]]
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )
    return holdout


def build_exclusion_map(
    interactions_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, set[str]]:
    positives = (
        interactions_df[interactions_df["label"] == 1][["user_id", "offer_id", "timestamp"]]
        .copy()
        .sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "offer_id"], keep="last")
    )
    holdout_pairs = set(
        tuple(row)
        for row in holdout_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )

    exclusion_map: dict[str, set[str]] = {}
    for user_id, offer_id in positives[["user_id", "offer_id"]].itertuples(index=False):
        pair = (str(user_id), str(offer_id))
        if pair in holdout_pairs:
            continue
        exclusion_map.setdefault(str(user_id), set()).add(str(offer_id))
    return exclusion_map


def build_train_interactions(
    interactions_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> pd.DataFrame:
    positives = (
        interactions_df[interactions_df["label"] == 1].copy()
        .sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "offer_id"], keep="last")
    )
    holdout_pairs = set(
        tuple(row)
        for row in holdout_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )
    mask = positives.apply(
        lambda row: (str(row["user_id"]), str(row["offer_id"])) not in holdout_pairs,
        axis=1,
    )
    return positives.loc[mask].reset_index(drop=True)


def build_real_validation_figures(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    ordered = metrics_df.sort_values("ndcg_at_k", ascending=False).copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="model", y="ndcg_at_k", hue="model", legend=False, palette="crest")
    plt.xticks(rotation=20, ha="right")
    plt.title("Real-Data Validation: NDCG@K by Model")
    plt.tight_layout()
    ndcg_path = figures_dir / "real_validation_ndcg_by_model.png"
    plt.savefig(ndcg_path, dpi=160)
    plt.close()
    saved_paths.append(ndcg_path)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="model", y="map_at_k", hue="model", legend=False, palette="viridis")
    plt.xticks(rotation=20, ha="right")
    plt.title("Real-Data Validation: MAP@K by Model")
    plt.tight_layout()
    map_path = figures_dir / "real_validation_map_by_model.png"
    plt.savefig(map_path, dpi=160)
    plt.close()
    saved_paths.append(map_path)

    return saved_paths


def write_real_validation_report(
    dataset_stats: dict[str, Any],
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    best = metrics_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
    lines = [
        "# Real-Data Validation Report",
        "",
        "## Dataset",
        f"- Name: {dataset_stats['dataset_name']}",
        f"- Source: {dataset_stats['source_url']}",
        f"- Kaggle mirror: {dataset_stats.get('kaggle_url', 'n/a')}",
        (
            "- Filtering: "
            f"min_user_interactions={dataset_stats['min_user_interactions']}, "
            f"min_item_interactions={dataset_stats['min_item_interactions']}"
        ),
        f"- Users after filtering: {dataset_stats['n_users']}",
        f"- Items after filtering: {dataset_stats['n_items']}",
        f"- Positive interactions after filtering: {dataset_stats['n_positive_interactions']}",
        f"- Time range: {dataset_stats['timestamp_min']} .. {dataset_stats['timestamp_max']}",
        "",
        "## Best Model",
        f"- Best by NDCG@{int(best['k'])}: `{best['model']}`",
        f"- NDCG@{int(best['k'])} = {best['ndcg_at_k']:.4f}",
        f"- MAP@{int(best['k'])} = {best['map_at_k']:.4f}",
        "",
        "## Metrics",
        "| model | precision@k | recall@k | map@k | ndcg@k |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in metrics_df.sort_values("ndcg_at_k", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.precision_at_k:.4f} | {row.recall_at_k:.4f} | {row.map_at_k:.4f} | {row.ndcg_at_k:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "- The benchmark now uses real transactional purchases instead of a media-rating dataset, which is a closer validation setting for recommendation from behavioral signals.",
        "- The domain is still retail, not banking, so results should be presented as proof that the pipeline transfers to real transaction logs rather than as a direct estimate of banking uplift.",
        "- The benchmark now includes both a nonlinear neural baseline (Neural CF) and an implemented graph-based SOTA branch (LightGCN), which makes the validation less dependent on purely linear baselines.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "movielens":
        dataset_path = download_movielens_100k(args.data_dir, force_download=args.force_download)
        interactions_df, offers_df, dataset_stats = load_movielens_implicit(
            dataset_dir=dataset_path,
            min_rating=args.min_rating,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
        interactions_output_name = "movielens_positive_interactions.csv"
        items_output_name = "movielens_items.csv"
    else:
        dataset_path = download_online_retail_xlsx(args.data_dir, force_download=args.force_download)
        interactions_df, offers_df, dataset_stats = load_online_retail_implicit(
            dataset_path=dataset_path,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            min_purchase_value=args.min_purchase_value,
            max_users=args.max_users if args.max_users > 0 else None,
            max_items=args.max_items if args.max_items > 0 else None,
        )
        interactions_output_name = "online_retail_positive_interactions.csv"
        items_output_name = "online_retail_items.csv"

    interactions_df.to_csv(args.output_dir / interactions_output_name, index=False)
    offers_df.to_csv(args.output_dir / items_output_name, index=False)

    holdout_df = build_holdout(interactions_df)
    train_interactions_df = build_train_interactions(interactions_df, holdout_df)
    exclusion_map = build_exclusion_map(interactions_df, holdout_df)
    user_ids = sorted(holdout_df["user_id"].astype(str).unique().tolist())
    ground_truth_df = holdout_df[["user_id", "offer_id"]].copy()

    metric_rows: list[dict[str, Any]] = []
    per_user_rows: list[pd.DataFrame] = []

    popularity_model = PopularityRecommender().fit(train_interactions_df)
    popularity_recs = popularity_model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    popularity_metrics, popularity_per_user = evaluate_ranking(popularity_recs, ground_truth_df, args.top_k)
    metric_rows.append({"model": "popularity", **popularity_metrics})
    popularity_per_user["model"] = "popularity"
    per_user_rows.append(popularity_per_user)

    item_knn_model = ImplicitItemKNNRecommender(n_neighbors=args.n_neighbors).fit(train_interactions_df)
    item_knn_recs = item_knn_model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    item_knn_metrics, item_knn_per_user = evaluate_ranking(item_knn_recs, ground_truth_df, args.top_k)
    metric_rows.append({"model": "item_knn", **item_knn_metrics, "n_neighbors": int(args.n_neighbors)})
    item_knn_per_user["model"] = "item_knn"
    per_user_rows.append(item_knn_per_user)

    mf_model = ImplicitMFRecommender(n_factors=args.n_factors, random_state=42).fit(train_interactions_df)
    mf_recs = mf_model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    mf_metrics, mf_per_user = evaluate_ranking(mf_recs, ground_truth_df, args.top_k)
    metric_rows.append({"model": "implicit_mf", **mf_metrics, "n_factors": int(args.n_factors)})
    mf_per_user["model"] = "implicit_mf"
    per_user_rows.append(mf_per_user)

    ncf_config = NeuralCFConfig(
        embedding_dim=int(args.ncf_embedding_dim),
        hidden_dims=parse_hidden_dims(args.ncf_hidden_dims),
        learning_rate=float(args.ncf_learning_rate),
        epochs=int(args.ncf_epochs),
        batch_size=int(args.ncf_batch_size),
        negative_samples=int(args.ncf_negative_samples),
        random_state=42,
    )
    ncf_model = NeuralCFRecommender(config=ncf_config).fit(train_interactions_df)
    ncf_recs = ncf_model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    ncf_metrics, ncf_per_user = evaluate_ranking(ncf_recs, ground_truth_df, args.top_k)
    metric_rows.append(
        {
            "model": "neural_cf",
            **ncf_metrics,
            "embedding_dim": int(ncf_config.embedding_dim),
            "hidden_dims": "|".join(str(value) for value in ncf_config.hidden_dims),
            "negative_samples": int(ncf_config.negative_samples),
        }
    )
    ncf_per_user["model"] = "neural_cf"
    per_user_rows.append(ncf_per_user)

    lightgcn_config = LightGCNConfig(
        embedding_dim=int(args.lightgcn_embedding_dim),
        n_layers=int(args.lightgcn_layers),
        learning_rate=float(args.lightgcn_learning_rate),
        epochs=int(args.lightgcn_epochs),
        batch_size=int(args.lightgcn_batch_size),
        samples_per_epoch=int(args.lightgcn_samples_per_epoch),
        l2_reg=float(args.lightgcn_l2_reg),
        random_state=42,
    )
    lightgcn_model = LightGCNRecommender(config=lightgcn_config).fit(train_interactions_df)
    lightgcn_recs = lightgcn_model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    lightgcn_metrics, lightgcn_per_user = evaluate_ranking(lightgcn_recs, ground_truth_df, args.top_k)
    metric_rows.append(
        {
            "model": "lightgcn",
            **lightgcn_metrics,
            "embedding_dim": int(lightgcn_config.embedding_dim),
            "n_layers": int(lightgcn_config.n_layers),
            "samples_per_epoch": int(lightgcn_config.samples_per_epoch),
        }
    )
    lightgcn_per_user["model"] = "lightgcn"
    per_user_rows.append(lightgcn_per_user)

    metrics_df = pd.DataFrame(metric_rows).sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)
    per_user_df = pd.concat(per_user_rows, ignore_index=True)

    metrics_df.to_csv(args.output_dir / "real_validation_metrics.csv", index=False)
    per_user_df.to_csv(args.output_dir / "real_validation_per_user_metrics.csv", index=False)
    holdout_df.to_csv(args.output_dir / "real_validation_holdout.csv", index=False)

    dataset_summary_path = args.output_dir / "real_validation_dataset_summary.json"
    dataset_summary_path.write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_paths = build_real_validation_figures(metrics_df, args.output_dir)

    report_path = write_real_validation_report(
        dataset_stats=dataset_stats,
        metrics_df=metrics_df,
        output_path=args.output_dir / "real_validation_report.md",
    )

    print("[ok] real-data validation finished")
    print(f"[ok] dataset path: {dataset_path}")
    print(f"[ok] users: {dataset_stats['n_users']}")
    print(f"[ok] items: {dataset_stats['n_items']}")
    print(f"[ok] positives: {dataset_stats['n_positive_interactions']}")
    print(f"[ok] metrics: {args.output_dir / 'real_validation_metrics.csv'}")
    print(f"[ok] dataset summary: {dataset_summary_path}")
    print(f"[ok] report: {report_path}")
    for figure_path in figure_paths:
        print(f"[ok] figure: {figure_path}")


if __name__ == "__main__":
    main()
