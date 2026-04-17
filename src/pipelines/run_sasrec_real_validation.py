from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REAL_DATA_DIR, REPORTS_DIR
from src.data.load_online_retail import download_online_retail_xlsx, load_online_retail_implicit
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.sasrec_recommender import SASRecConfig, SASRecRecommender
from src.pipelines.run_real_dataset_validation import build_exclusion_map, build_holdout, build_train_interactions


def write_report(metrics: dict[str, float | int], output_path: Path) -> Path:
    lines = [
        "# SASRec Real-Data Validation Report",
        "",
        "## Configuration",
        f"- embedding_dim: {metrics['embedding_dim']}",
        f"- num_heads: {metrics['num_heads']}",
        f"- num_blocks: {metrics['num_blocks']}",
        f"- max_seq_len: {metrics['max_seq_len']}",
        f"- window_stride: {metrics['window_stride']}",
        f"- dropout: {metrics['dropout']}",
        f"- learning_rate: {metrics['learning_rate']}",
        f"- epochs: {metrics['epochs']}",
        f"- batch_size: {metrics['batch_size']}",
        f"- samples_per_epoch: {metrics['samples_per_epoch']}",
        "",
        "## Dataset",
        f"- users: {metrics['dataset_users']}",
        f"- items: {metrics['dataset_items']}",
        f"- positives: {metrics['dataset_positives']}",
        "",
        "## Metrics",
        f"- Precision@{metrics['k']}: {metrics['precision_at_k']:.4f}",
        f"- Recall@{metrics['k']}: {metrics['recall_at_k']:.4f}",
        f"- MAP@{metrics['k']}: {metrics['map_at_k']:.4f}",
        f"- NDCG@{metrics['k']}: {metrics['ndcg_at_k']:.4f}",
        f"- Final training loss: {metrics['training_loss_final']:.4f}",
        "",
        "## Interpretation",
        "- This run checks a true sequence-aware branch on the real transaction log rather than on the synthetic offer benchmark.",
        "- In the current implementation and split, SASRec underperforms the strongest non-sequential baselines, so it should be presented as an implemented next-stage model rather than as the new leader.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SASRec validation on the real transaction dataset.")
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
        help="Directory where SASRec artifacts are stored.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations for evaluation.")
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
    parser.add_argument("--max-users", type=int, default=3000, help="Optional upper bound on retained users.")
    parser.add_argument("--max-items", type=int, default=1500, help="Optional upper bound on retained items.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension for SASRec.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-blocks", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Maximum sequence length.")
    parser.add_argument("--window-stride", type=int, default=1, help="Stride over prefix targets.")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch count.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=50000,
        help="Optional cap on sampled prefix-target pairs per epoch.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if it already exists locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = download_online_retail_xlsx(args.data_dir, force_download=args.force_download)
    interactions_df, offers_df, dataset_stats = load_online_retail_implicit(
        dataset_path=dataset_path,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        min_purchase_value=args.min_purchase_value,
        max_users=args.max_users if args.max_users > 0 else None,
        max_items=args.max_items if args.max_items > 0 else None,
    )

    holdout_df = build_holdout(interactions_df)
    train_interactions_df = build_train_interactions(interactions_df, holdout_df)
    exclusion_map = build_exclusion_map(interactions_df, holdout_df)
    user_ids = sorted(holdout_df["user_id"].astype(str).unique().tolist())
    ground_truth_df = holdout_df[["user_id", "offer_id"]].copy()

    config = SASRecConfig(
        embedding_dim=int(args.embedding_dim),
        num_heads=int(args.num_heads),
        num_blocks=int(args.num_blocks),
        dropout=float(args.dropout),
        max_seq_len=int(args.max_seq_len),
        window_stride=int(args.window_stride),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        samples_per_epoch=int(args.samples_per_epoch),
        random_state=42,
    )

    model = SASRecRecommender(config=config).fit(train_interactions_df)
    recommendations_df = model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )

    metrics, per_user_df = evaluate_ranking(
        recommendations_df=recommendations_df,
        ground_truth_df=ground_truth_df,
        k=args.top_k,
    )
    metrics.update(
        {
            "embedding_dim": int(config.embedding_dim),
            "num_heads": int(config.num_heads),
            "num_blocks": int(config.num_blocks),
            "max_seq_len": int(config.max_seq_len),
            "window_stride": int(config.window_stride),
            "dropout": float(config.dropout),
            "learning_rate": float(config.learning_rate),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "samples_per_epoch": int(config.samples_per_epoch),
            "training_loss_final": float(model.loss_history_[-1]) if model.loss_history_ else None,
            "dataset_users": int(dataset_stats["n_users"]),
            "dataset_items": int(dataset_stats["n_items"]),
            "dataset_positives": int(dataset_stats["n_positive_interactions"]),
        }
    )

    with (args.output_dir / "sasrec_real_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "sasrec_real_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "sasrec_real_per_user_metrics.csv", index=False)
    holdout_df.to_csv(args.output_dir / "sasrec_real_holdout.csv", index=False)
    offers_df.to_csv(args.output_dir / "sasrec_real_items.csv", index=False)
    pd.DataFrame([dataset_stats]).to_json(
        args.output_dir / "sasrec_real_dataset_summary.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    report_path = write_report(metrics, args.output_dir / "sasrec_real_report.md")

    print("[ok] sasrec real-data validation finished")
    print(f"[ok] users: {dataset_stats['n_users']}")
    print(f"[ok] items: {dataset_stats['n_items']}")
    print(f"[ok] positives: {dataset_stats['n_positive_interactions']}")
    print(f"[ok] report: {report_path}")
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
