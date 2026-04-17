from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR
from src.data.build_offer_catalog import build_offer_catalog
from src.data.generate_synthetic_transactions import generate_transactions, generate_users
from src.data.simulate_interactions import simulate_interactions
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.baseline_recommender import TransactionSimilarityRecommender
from src.models.hybrid_semantic_recommender import HybridSemanticRecommender
from src.models.implicit_mf_recommender import ImplicitMFRecommender
from src.models.item_knn_recommender import ImplicitItemKNNRecommender
from src.models.time_decay_recommender import TimeDecayRecommender
from src.pipelines.run_baseline_pipeline import build_exclusion_map, build_test_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark for recommendation models.")
    parser.add_argument("--n-users", type=int, default=800)
    parser.add_argument("--avg-transactions", type=int, default=140)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,13,21,42,77",
        help="Comma-separated seeds, e.g. '7,13,21,42,77'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "multiseed",
        help="Directory for multi-seed outputs.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    return parser.parse_args()


def build_train_interactions(
    interactions_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> pd.DataFrame:
    holdout_pairs = set(
        tuple(x) for x in holdout_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )
    mask = interactions_df.apply(
        lambda r: (str(r["user_id"]), str(r["offer_id"])) not in holdout_pairs,
        axis=1,
    )
    return interactions_df.loc[mask].reset_index(drop=True)


def _bootstrap_mean_diff(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "ci_2_5": 0.0,
            "ci_97_5": 0.0,
            "prob_diff_gt_zero": 0.0,
        }

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = values[idx].mean(axis=1)
    return {
        "mean_diff": float(values.mean()),
        "ci_2_5": float(np.percentile(boot, 2.5)),
        "ci_97_5": float(np.percentile(boot, 97.5)),
        "prob_diff_gt_zero": float((boot > 0).mean()),
    }


def evaluate_models_for_seed(
    seed: int,
    n_users: int,
    avg_transactions: int,
    months: int,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    users_df = generate_users(n_users=n_users, rng=rng)
    transactions_df = generate_transactions(
        users_df=users_df,
        avg_transactions=avg_transactions,
        months=months,
        rng=rng,
    )
    offers_df = build_offer_catalog()
    interactions_df = simulate_interactions(users_df, offers_df, seed=seed + 1)

    holdout_df = build_test_ground_truth(interactions_df)
    exclusion_map = build_exclusion_map(interactions_df, holdout_df)
    user_ids = sorted(holdout_df["user_id"].astype(str).unique().tolist())
    gt = holdout_df[["user_id", "offer_id"]]

    rows: list[dict[str, float | int | str]] = []
    user_diffs_rows: list[dict[str, float | int | str]] = []

    # Model 1: profile baseline.
    profile_model = TransactionSimilarityRecommender().fit(transactions_df, offers_df)
    profile_recs = profile_model.recommend_for_users(
        user_ids=user_ids,
        top_k=top_k,
        exclude_by_user=exclusion_map,
    )
    profile_metrics, profile_per_user = evaluate_ranking(profile_recs, gt, top_k)
    rows.append({"seed": seed, "model": "profile_baseline", **profile_metrics})

    # Model 2: item-kNN collaborative filtering.
    train_interactions_df = build_train_interactions(interactions_df, holdout_df)
    item_knn_model = ImplicitItemKNNRecommender(n_neighbors=10).fit(train_interactions_df)
    item_knn_recs = item_knn_model.recommend_for_users(
        user_ids=user_ids,
        top_k=top_k,
        exclude_by_user=exclusion_map,
    )
    item_knn_metrics, _ = evaluate_ranking(item_knn_recs, gt, top_k)
    rows.append({"seed": seed, "model": "item_knn", **item_knn_metrics})

    # Model 3: implicit MF.
    mf_model = ImplicitMFRecommender(n_factors=10, random_state=seed).fit(train_interactions_df)
    mf_recs = mf_model.recommend_for_users(
        user_ids=user_ids,
        top_k=top_k,
        exclude_by_user=exclusion_map,
    )
    mf_metrics, _ = evaluate_ranking(mf_recs, gt, top_k)
    rows.append({"seed": seed, "model": "implicit_mf", **mf_metrics})

    # Model 4: hybrid semantic.
    hybrid_model = HybridSemanticRecommender(
        profile_weight=0.7,
        semantic_weight=0.3,
    ).fit(transactions_df, offers_df)
    hybrid_recs = hybrid_model.recommend_for_users(
        user_ids=user_ids,
        top_k=top_k,
        exclude_by_user=exclusion_map,
    )
    hybrid_metrics, _ = evaluate_ranking(hybrid_recs, gt, top_k)
    rows.append({"seed": seed, "model": "hybrid_semantic", **hybrid_metrics})

    # Model 5: tuned time-decay.
    td_model = TimeDecayRecommender(
        decay_rate=0.01,
        short_term_days=30,
        short_term_weight=0.2,
        spend_weight=0.6,
        freq_weight=0.4,
    ).fit(transactions_df, offers_df)
    td_recs = td_model.recommend_for_users(
        user_ids=user_ids,
        top_k=top_k,
        exclude_by_user=exclusion_map,
    )
    td_metrics, td_per_user = evaluate_ranking(td_recs, gt, top_k)
    rows.append({"seed": seed, "model": "time_decay", **td_metrics})

    merged = td_per_user.merge(
        profile_per_user,
        on="user_id",
        suffixes=("_time_decay", "_profile_baseline"),
    )
    merged["seed"] = seed
    merged["delta_ndcg_at_k"] = merged["ndcg_at_k_time_decay"] - merged["ndcg_at_k_profile_baseline"]
    merged["delta_map_at_k"] = merged["map_at_k_time_decay"] - merged["map_at_k_profile_baseline"]
    merged["delta_recall_at_k"] = merged["recall_at_k_time_decay"] - merged["recall_at_k_profile_baseline"]
    merged["delta_precision_at_k"] = (
        merged["precision_at_k_time_decay"] - merged["precision_at_k_profile_baseline"]
    )

    for r in merged.itertuples(index=False):
        user_diffs_rows.append(
            {
                "seed": int(r.seed),
                "user_id": str(r.user_id),
                "delta_ndcg_at_k": float(r.delta_ndcg_at_k),
                "delta_map_at_k": float(r.delta_map_at_k),
                "delta_recall_at_k": float(r.delta_recall_at_k),
                "delta_precision_at_k": float(r.delta_precision_at_k),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(user_diffs_rows)


def build_markdown_summary(
    summary_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    bootstrap: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    best = summary_df.sort_values("mean_ndcg_at_k", ascending=False).iloc[0]
    lines = [
        "# Multi-Seed Benchmark Summary",
        "",
        "## Setup",
        f"- Seeds: {', '.join(str(x) for x in sorted(seed_df['seed'].unique().tolist()))}",
        f"- Users per seed: {int(seed_df['n_users_evaluated'].max())}",
        f"- Models: {', '.join(summary_df['model'].tolist())}",
        "",
        "## Best Model (by mean NDCG@K)",
        f"- {best['model']}",
        f"- mean NDCG@K = {best['mean_ndcg_at_k']:.4f} (std {best['std_ndcg_at_k']:.4f})",
        f"- mean MAP@K = {best['mean_map_at_k']:.4f} (std {best['std_map_at_k']:.4f})",
        "",
        "## Mean Metrics by Model",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            (
                f"- {row.model}: "
                f"NDCG@K={row.mean_ndcg_at_k:.4f} (+/- {row.std_ndcg_at_k:.4f}), "
                f"MAP@K={row.mean_map_at_k:.4f} (+/- {row.std_map_at_k:.4f}), "
                f"Recall@K={row.mean_recall_at_k:.4f} (+/- {row.std_recall_at_k:.4f})"
            )
        )

    lines += [
        "",
        "## Time-Decay vs Profile Baseline (pooled user-level diffs across seeds)",
    ]

    for metric in ["delta_ndcg_at_k", "delta_map_at_k", "delta_recall_at_k", "delta_precision_at_k"]:
        b = bootstrap[metric]
        lines.append(
            (
                f"- {metric}: mean={b['mean_diff']:.6f}, "
                f"95% CI=[{b['ci_2_5']:.6f}, {b['ci_97_5']:.6f}], "
                f"P(diff>0)={b['prob_diff_gt_zero']:.3f}"
            )
        )

    lines += [
        "",
        "## Notes",
        "- This benchmark measures stability of conclusions across random seeds.",
        "- If CIs still include zero, gains should be reported as directional rather than definitive.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_figures(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Mean NDCG@K with std error bars.
    ordered = summary_df.sort_values("mean_ndcg_at_k", ascending=False).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=ordered,
        x="model",
        y="mean_ndcg_at_k",
        hue="model",
        legend=False,
        palette="crest",
    )
    plt.errorbar(
        x=np.arange(len(ordered)),
        y=ordered["mean_ndcg_at_k"],
        yerr=ordered["std_ndcg_at_k"],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Multi-Seed Mean NDCG@K (with Std)")
    plt.tight_layout()
    p1 = fig_dir / "multiseed_mean_ndcg_with_std.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    saved.append(p1)

    # Mean MAP@K with std error bars.
    ordered_map = summary_df.sort_values("mean_map_at_k", ascending=False).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=ordered_map,
        x="model",
        y="mean_map_at_k",
        hue="model",
        legend=False,
        palette="viridis",
    )
    plt.errorbar(
        x=np.arange(len(ordered_map)),
        y=ordered_map["mean_map_at_k"],
        yerr=ordered_map["std_map_at_k"],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Multi-Seed Mean MAP@K (with Std)")
    plt.tight_layout()
    p2 = fig_dir / "multiseed_mean_map_with_std.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    saved.append(p2)

    return saved


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metric_rows: list[pd.DataFrame] = []
    diff_rows: list[pd.DataFrame] = []

    for seed in seeds:
        metrics_df, diffs_df = evaluate_models_for_seed(
            seed=seed,
            n_users=args.n_users,
            avg_transactions=args.avg_transactions,
            months=args.months,
            top_k=args.top_k,
        )
        metric_rows.append(metrics_df)
        diff_rows.append(diffs_df)
        print(f"[ok] seed {seed} finished")

    seed_df = pd.concat(metric_rows, ignore_index=True)
    diffs_df = pd.concat(diff_rows, ignore_index=True)
    seed_df.to_csv(args.output_dir / "multiseed_seed_metrics.csv", index=False)
    diffs_df.to_csv(args.output_dir / "multiseed_user_deltas_time_decay_vs_baseline.csv", index=False)

    summary_df = (
        seed_df.groupby("model")[["precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "model",
        "mean_precision_at_k",
        "std_precision_at_k",
        "mean_recall_at_k",
        "std_recall_at_k",
        "mean_map_at_k",
        "std_map_at_k",
        "mean_ndcg_at_k",
        "std_ndcg_at_k",
    ]
    summary_df = summary_df.sort_values("mean_ndcg_at_k", ascending=False)
    summary_df.to_csv(args.output_dir / "multiseed_summary_metrics.csv", index=False)

    rng = np.random.default_rng(42)
    bootstrap = {
        "delta_ndcg_at_k": _bootstrap_mean_diff(
            diffs_df["delta_ndcg_at_k"].to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "delta_map_at_k": _bootstrap_mean_diff(
            diffs_df["delta_map_at_k"].to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "delta_recall_at_k": _bootstrap_mean_diff(
            diffs_df["delta_recall_at_k"].to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "delta_precision_at_k": _bootstrap_mean_diff(
            diffs_df["delta_precision_at_k"].to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
    }

    bootstrap_path = args.output_dir / "multiseed_bootstrap_time_decay_vs_baseline.json"
    bootstrap_path.write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = args.output_dir / "multiseed_summary_report.md"
    build_markdown_summary(summary_df, seed_df, bootstrap, markdown_path)
    fig_paths = save_figures(summary_df, args.output_dir)

    print("[ok] multi-seed benchmark completed")
    print(f"[ok] seed metrics: {args.output_dir / 'multiseed_seed_metrics.csv'}")
    print(f"[ok] summary: {args.output_dir / 'multiseed_summary_metrics.csv'}")
    print(f"[ok] bootstrap: {bootstrap_path}")
    print(f"[ok] report: {markdown_path}")
    for p in fig_paths:
        print(f"[ok] figure: {p}")


if __name__ == "__main__":
    main()
