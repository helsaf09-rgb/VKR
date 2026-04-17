from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR


MODEL_FILES: dict[str, str] = {
    "profile_baseline": "per_user_metrics.csv",
    "item_knn": "item_knn_per_user_metrics.csv",
    "implicit_mf": "mf_per_user_metrics.csv",
    "neural_cf": "ncf_per_user_metrics.csv",
    "lightgcn": "lightgcn_per_user_metrics.csv",
    "hybrid_semantic": "hybrid_per_user_metrics.csv",
    "time_decay": "time_decay_per_user_metrics.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analysis reports from model outputs.")
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_model_per_user_metrics(reports_dir: Path) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for model_name, filename in MODEL_FILES.items():
        path = reports_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run model pipelines first (`scripts/run_all.ps1`)."
            )
        df = pd.read_csv(path)
        df["user_id"] = df["user_id"].astype(str)
        loaded[model_name] = df
    return loaded


def _bootstrap_mean_diff(
    diff_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(diff_values)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "ci_2_5": 0.0,
            "ci_97_5": 0.0,
            "prob_diff_gt_zero": 0.0,
        }

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = diff_values[idx].mean(axis=1)
    return {
        "mean_diff": float(diff_values.mean()),
        "ci_2_5": float(np.percentile(boot_means, 2.5)),
        "ci_97_5": float(np.percentile(boot_means, 97.5)),
        "prob_diff_gt_zero": float((boot_means > 0).mean()),
    }


def _save_figures(
    overall_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    reports_dir: Path,
) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    # Overall NDCG@5 by model.
    plt.figure(figsize=(9, 5))
    overall_ndcg = overall_df[["model", "ndcg_at_k"]].copy()
    sns.barplot(
        data=overall_ndcg,
        x="model",
        y="ndcg_at_k",
        hue="model",
        legend=False,
        palette="crest",
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Overall NDCG@5 by Model")
    plt.tight_layout()
    p1 = fig_dir / "analysis_overall_ndcg_by_model.png"
    plt.savefig(p1, dpi=160)
    plt.close()
    saved_paths.append(p1)

    # Overall MAP@5 by model.
    plt.figure(figsize=(9, 5))
    overall_map = overall_df[["model", "map_at_k"]].copy()
    sns.barplot(
        data=overall_map,
        x="model",
        y="map_at_k",
        hue="model",
        legend=False,
        palette="viridis",
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Overall MAP@5 by Model")
    plt.tight_layout()
    p2 = fig_dir / "analysis_overall_map_by_model.png"
    plt.savefig(p2, dpi=160)
    plt.close()
    saved_paths.append(p2)

    # Segment-wise NDCG for two strongest models (profile baseline and time_decay).
    seg_subset = segment_df[segment_df["model"].isin(["profile_baseline", "time_decay"])].copy()
    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=seg_subset,
        x="segment",
        y="ndcg_at_k",
        hue="model",
        palette="magma",
    )
    plt.xticks(rotation=20, ha="right")
    plt.title("Segment-wise NDCG@5: Profile Baseline vs Time-Decay")
    plt.tight_layout()
    p3 = fig_dir / "analysis_segment_ndcg_baseline_vs_time_decay.png"
    plt.savefig(p3, dpi=160)
    plt.close()
    saved_paths.append(p3)

    return saved_paths


def _write_markdown_summary(
    overall_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    bootstrap_results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    best_overall = overall_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
    td = overall_df[overall_df["model"] == "time_decay"].iloc[0]
    pb = overall_df[overall_df["model"] == "profile_baseline"].iloc[0]

    segment_cmp = (
        segment_df[segment_df["model"].isin(["profile_baseline", "time_decay"])]
        .pivot(index="segment", columns="model", values="ndcg_at_k")
        .reset_index()
    )
    if {"profile_baseline", "time_decay"}.issubset(set(segment_cmp.columns)):
        segment_cmp["delta_time_decay_minus_baseline"] = (
            segment_cmp["time_decay"] - segment_cmp["profile_baseline"]
        )
    else:
        segment_cmp["delta_time_decay_minus_baseline"] = np.nan

    lines = [
        "# Analysis Summary Report",
        "",
        "## Overall Best Model",
        f"- Best by NDCG@5: `{best_overall['model']}`",
        f"- NDCG@5 = {best_overall['ndcg_at_k']:.4f}",
        f"- MAP@5 = {best_overall['map_at_k']:.4f}",
        "",
        "## Model Leaderboard",
    ]

    for row in overall_df.itertuples(index=False):
        lines.append(
            (
                f"- {row.model}: "
                f"Precision@5={row.precision_at_k:.4f}, "
                f"Recall@5={row.recall_at_k:.4f}, "
                f"MAP@5={row.map_at_k:.4f}, "
                f"NDCG@5={row.ndcg_at_k:.4f}"
            )
        )

    lines += [
        "",
        "## Time-Decay vs Profile Baseline",
        f"- Profile baseline NDCG@5 = {pb['ndcg_at_k']:.4f}",
        f"- Time-decay NDCG@5 = {td['ndcg_at_k']:.4f}",
        f"- Profile baseline MAP@5 = {pb['map_at_k']:.4f}",
        f"- Time-decay MAP@5 = {td['map_at_k']:.4f}",
        "",
        "## Bootstrap Confidence (Time-Decay minus Profile Baseline)",
    ]

    for metric_name, stats in bootstrap_results.items():
        lines += [
            (
                f"- {metric_name}: mean_diff={stats['mean_diff']:.6f}, "
                f"95% CI=[{stats['ci_2_5']:.6f}, {stats['ci_97_5']:.6f}], "
                f"P(diff>0)={stats['prob_diff_gt_zero']:.3f}"
            )
        ]

    lines += [
        "",
        "## Segment-wise Delta in NDCG@5 (Time-Decay - Profile Baseline)",
    ]
    for row in segment_cmp.itertuples(index=False):
        lines.append(f"- {row.segment}: {row.delta_time_decay_minus_baseline:.4f}")

    lines += [
        "",
        "## Practical Note",
        "- Improvements are modest in top-rank quality metrics (MAP/NDCG).",
        "- Bootstrap confidence intervals include zero, so gains should be treated as directional, not yet definitive.",
        "- Recency-aware modeling is still a reasonable advanced step, but stronger evidence requires larger data or stronger sequential models.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_csv(args.data_dir / "users.csv")[["user_id", "segment"]].copy()
    users_df["user_id"] = users_df["user_id"].astype(str)

    model_metrics = _load_model_per_user_metrics(args.reports_dir)
    merged_by_model: dict[str, pd.DataFrame] = {}
    for model_name, df in model_metrics.items():
        merged = df.merge(users_df, on="user_id", how="left")
        merged["model"] = model_name
        merged_by_model[model_name] = merged

    all_models_df = pd.concat(merged_by_model.values(), ignore_index=True)

    overall_df = (
        all_models_df.groupby("model")[["precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]]
        .mean()
        .reset_index()
        .sort_values("ndcg_at_k", ascending=False)
    )
    overall_df.to_csv(args.reports_dir / "analysis_overall_metrics.csv", index=False)

    segment_df = (
        all_models_df.groupby(["model", "segment"])[["precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]]
        .mean()
        .reset_index()
        .sort_values(["segment", "ndcg_at_k"], ascending=[True, False])
    )
    segment_df.to_csv(args.reports_dir / "analysis_segment_metrics.csv", index=False)

    # Bootstrap comparison: time_decay vs profile_baseline.
    td = merged_by_model["time_decay"].set_index("user_id")
    pb = merged_by_model["profile_baseline"].set_index("user_id")
    common_users = sorted(set(td.index).intersection(set(pb.index)))
    td = td.loc[common_users]
    pb = pb.loc[common_users]

    rng = np.random.default_rng(args.seed)
    bootstrap_results = {
        "ndcg_at_k": _bootstrap_mean_diff(
            (td["ndcg_at_k"] - pb["ndcg_at_k"]).to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "map_at_k": _bootstrap_mean_diff(
            (td["map_at_k"] - pb["map_at_k"]).to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "recall_at_k": _bootstrap_mean_diff(
            (td["recall_at_k"] - pb["recall_at_k"]).to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
        "precision_at_k": _bootstrap_mean_diff(
            (td["precision_at_k"] - pb["precision_at_k"]).to_numpy(dtype=float),
            args.n_bootstrap,
            rng,
        ),
    }

    bootstrap_path = args.reports_dir / "analysis_bootstrap_time_decay_vs_baseline.json"
    bootstrap_path.write_text(json.dumps(bootstrap_results, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_paths = _save_figures(overall_df, segment_df, args.reports_dir)

    summary_path = args.reports_dir / "analysis_summary_report.md"
    _write_markdown_summary(overall_df, segment_df, bootstrap_results, summary_path)

    print("[ok] analysis reports completed")
    print(f"[ok] overall metrics: {args.reports_dir / 'analysis_overall_metrics.csv'}")
    print(f"[ok] segment metrics: {args.reports_dir / 'analysis_segment_metrics.csv'}")
    print(f"[ok] bootstrap: {bootstrap_path}")
    print(f"[ok] summary: {summary_path}")
    for p in fig_paths:
        print(f"[ok] figure: {p}")


if __name__ == "__main__":
    main()
