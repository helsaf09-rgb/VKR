# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "thesis_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def wrap_text_lines(text: str, width: int, max_lines: int) -> str:
    wrapped: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            wrapped.append("")
            continue
        chunks = textwrap.wrap(
            line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped.extend(chunks or [""])
    return "\n".join(wrapped[:max_lines])


def render_text_card(
    output_path: Path,
    title: str,
    body: str,
    *,
    subtitle: str | None = None,
    mono: bool = False,
    width: int = 98,
    max_lines: int = 34,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.94,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#1f2937",
            facecolor="white",
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.9),
            0.96,
            0.07,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0,
            facecolor="#e5eef7",
        )
    )
    ax.text(
        0.05,
        0.935,
        title,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111827",
        parse_math=False,
    )
    if subtitle:
        ax.text(0.95, 0.935, subtitle, fontsize=10, ha="right", va="center", color="#4b5563", parse_math=False)

    wrapped = wrap_text_lines(body, width=width, max_lines=max_lines)
    ax.text(
        0.05,
        0.87,
        wrapped,
        ha="left",
        va="top",
        family="DejaVu Sans Mono" if mono else "DejaVu Serif",
        fontsize=10.5 if mono else 11,
        color="#111827",
        linespacing=1.35,
        parse_math=False,
    )
    save(fig, output_path)


def extract_text(path: Path, *, max_lines: int = 40) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    clipped = [line.expandtabs(2)[:110] for line in lines[:max_lines]]
    return "\n".join(clipped)


def build_project_tree(root: Path, *, max_depth: int = 3, max_entries_per_dir: int = 8) -> str:
    ignore = {".git", ".venv", "tmp", "__pycache__", ".ipynb_checkpoints"}

    def walk(path: Path, prefix: str = "", depth: int = 0) -> list[str]:
        if depth > max_depth:
            return []
        entries = [p for p in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())) if p.name not in ignore]
        lines: list[str] = []
        visible = entries[:max_entries_per_dir]
        for idx, entry in enumerate(visible):
            branch = "└── " if idx == len(visible) - 1 else "├── "
            lines.append(f"{prefix}{branch}{entry.name}")
            if entry.is_dir():
                extension = "    " if idx == len(visible) - 1 else "│   "
                lines.extend(walk(entry, prefix + extension, depth + 1))
        if len(entries) > max_entries_per_dir:
            hidden = len(entries) - max_entries_per_dir
            lines.append(f"{prefix}└── ... ({hidden} more items)")
        return lines

    return "\n".join([root.name, *walk(root)])


def draw_box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#111827")


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="#374151"),
    )


def generate_pipeline_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_box(ax, 0.04, 0.42, 0.16, 0.2, "Синтетические\nтранзакции", "#dbeafe")
    draw_box(ax, 0.24, 0.42, 0.16, 0.2, "Инженерия\nпризнаков", "#e0f2fe")
    draw_box(ax, 0.44, 0.42, 0.16, 0.2, "Слой моделей\nMF / NCF /\nLightGCN / SASRec", "#dcfce7")
    draw_box(ax, 0.64, 0.42, 0.14, 0.2, "Оценка\nметрик", "#fef3c7")
    draw_box(ax, 0.82, 0.42, 0.14, 0.2, "API и UI\nпрототип", "#fee2e2")
    for start, end in [(0.20, 0.24), (0.40, 0.44), (0.60, 0.64), (0.78, 0.82)]:
        draw_arrow(ax, start, 0.52, end, 0.52)
    ax.text(0.5, 0.82, "Сквозной пайплайн персонализации", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.22, "Генерация данных, моделирование, оценка и выдача рекомендаций собраны в едином воспроизводимом контуре.", ha="center", va="center", fontsize=11, color="#374151")
    save(fig, OUTPUT_DIR / "pipeline_architecture.png")


def generate_dataset_structure() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_box(ax, 0.09, 0.64, 0.25, 0.16, "users.csv\nuser_id, segment,\nпредпочтения по категориям", "#dbeafe")
    draw_box(ax, 0.39, 0.64, 0.25, 0.16, "transactions.csv\nuser_id, категория,\namount, timestamp", "#dcfce7")
    draw_box(ax, 0.69, 0.64, 0.22, 0.16, "offers.csv\noffer_id, тип продукта,\nтекстовое описание", "#fef3c7")
    draw_box(ax, 0.39, 0.30, 0.25, 0.16, "interactions.csv\nклики / открытия /\nположительный отклик", "#fee2e2")
    draw_arrow(ax, 0.22, 0.64, 0.47, 0.52)
    draw_arrow(ax, 0.52, 0.64, 0.52, 0.46)
    draw_arrow(ax, 0.80, 0.64, 0.57, 0.52)
    ax.text(0.5, 0.88, "Артефакты синтетического датасета", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.16, "Четыре таблицы разделяют сущности, сырые события, метаданные офферов и сигналы пользовательского отклика.", ha="center", va="center", fontsize=11, color="#374151")
    save(fig, OUTPUT_DIR / "synthetic_dataset_structure.png")


def generate_profile_scheme() -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_box(ax, 0.05, 0.38, 0.18, 0.22, "История\nтранзакций", "#dbeafe")
    draw_box(ax, 0.29, 0.38, 0.19, 0.22, "Взвешенный\nпрофиль пользователя", "#dcfce7")
    draw_box(ax, 0.54, 0.38, 0.19, 0.22, "Профиль оффера /\nпространство\nэмбеддингов", "#fef3c7")
    draw_box(ax, 0.79, 0.38, 0.16, 0.22, "Top-K\nрекомендации", "#fee2e2")
    draw_arrow(ax, 0.23, 0.49, 0.29, 0.49)
    draw_arrow(ax, 0.48, 0.49, 0.54, 0.49)
    draw_arrow(ax, 0.73, 0.49, 0.79, 0.49)
    ax.text(0.62, 0.67, "Сходство и итоговый скоринг", ha="center", va="center", fontsize=11, color="#374151")
    ax.text(0.5, 0.82, "Построение профиля и итоговый расчет рекомендаций", ha="center", va="center", fontsize=15, fontweight="bold")
    save(fig, OUTPUT_DIR / "profile_scoring_scheme.png")


def generate_synthetic_model_metrics() -> None:
    df = pd.read_csv(ROOT / "reports" / "analysis_overall_metrics.csv").sort_values("ndcg_at_k", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(df))
    ax.barh(y + 0.17, df["ndcg_at_k"], height=0.3, color="#2563eb", label="NDCG@5")
    ax.barh(y - 0.17, df["map_at_k"], height=0.3, color="#f59e0b", label="MAP@5")
    ax.set_yticks(y, df["model"])
    ax.set_xlabel("Значение метрики")
    ax.set_title("Сравнение NDCG@5 и MAP@5 на синтетическом наборе")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save(fig, OUTPUT_DIR / "synthetic_model_metrics.png")


def generate_multiseed_summary() -> None:
    df = pd.read_csv(ROOT / "reports" / "multiseed" / "multiseed_summary_metrics.csv").sort_values("mean_ndcg_at_k", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.barh(
        df["model"],
        df["mean_ndcg_at_k"],
        xerr=df["std_ndcg_at_k"],
        color="#0f766e",
        ecolor="#134e4a",
        capsize=4,
    )
    ax.set_title("Сводка multi-seed benchmark (средний NDCG@5 ± std)")
    ax.set_xlabel("Средний NDCG@5")
    ax.grid(axis="x", alpha=0.25)
    save(fig, OUTPUT_DIR / "multiseed_summary.png")


def generate_segment_distribution() -> None:
    users = pd.read_csv(ROOT / "data" / "synthetic" / "users.csv")
    counts = users["segment"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(counts.index, counts.values, color="#4f46e5")
    ax.set_title("Распределение пользователей по поведенческим сегментам")
    ax.set_ylabel("Число пользователей")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save(fig, OUTPUT_DIR / "segment_distribution.png")


def generate_segment_spend_summary() -> None:
    users = pd.read_csv(ROOT / "data" / "synthetic" / "users.csv", usecols=["user_id", "segment"])
    tx = pd.read_csv(ROOT / "data" / "synthetic" / "transactions.csv", usecols=["user_id", "amount"])
    totals = tx.groupby("user_id", as_index=False)["amount"].sum().merge(users, on="user_id", how="left")
    summary = totals.groupby("segment")["amount"].agg(mean_spend="mean", median_spend="median").sort_values("mean_spend", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["mean_spend"], width=0.36, color="#2563eb", label="Средний суммарный расход")
    ax.bar(x + 0.18, summary["median_spend"], width=0.36, color="#0f766e", label="Медианный суммарный расход")
    ax.set_xticks(x, summary.index, rotation=25)
    ax.set_ylabel("Сумма")
    ax.set_title("Сводка расходов по сегментам")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, OUTPUT_DIR / "segment_spend_summary.png")


def generate_time_decay_tuning() -> None:
    df = pd.read_csv(ROOT / "reports" / "time_decay_sweep_results.csv")
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    scatter = ax.scatter(
        df["short_term_weight"],
        df["ndcg_at_k"],
        c=df["decay_rate"],
        s=85,
        cmap="viridis",
        alpha=0.9,
    )
    top_rows = df.sort_values("ndcg_at_k", ascending=False).head(5)
    for _, row in top_rows.iterrows():
        ax.annotate(
            f"λ={row['decay_rate']:.3f}",
            (row["short_term_weight"], row["ndcg_at_k"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Вес краткосрочного профиля")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Подбор гиперпараметров time-decay модели")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Коэффициент затухания")
    ax.grid(alpha=0.25)
    save(fig, OUTPUT_DIR / "time_decay_tuning.png")


def generate_bootstrap_ci_chart() -> None:
    data = json.loads((ROOT / "reports" / "analysis_bootstrap_time_decay_vs_baseline.json").read_text(encoding="utf-8"))
    metrics = list(data.keys())
    mean_diff = [data[m]["mean_diff"] for m in metrics]
    lower = [data[m]["mean_diff"] - data[m]["ci_2_5"] for m in metrics]
    upper = [data[m]["ci_97_5"] - data[m]["mean_diff"] for m in metrics]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    y = np.arange(len(metrics))
    ax.errorbar(mean_diff, y, xerr=[lower, upper], fmt="o", color="#1d4ed8", ecolor="#111827", capsize=4)
    ax.axvline(0, linestyle="--", color="#b91c1c", linewidth=1)
    ax.set_yticks(y, metrics)
    ax.set_xlabel("Разность: time_decay - profile_baseline")
    ax.set_title("Bootstrap-интервалы для различий по метрикам")
    ax.grid(axis="x", alpha=0.25)
    save(fig, OUTPUT_DIR / "bootstrap_ci_chart.png")


def generate_segment_metrics() -> None:
    df = pd.read_csv(ROOT / "reports" / "analysis_segment_metrics.csv")
    subset = df[df["model"].isin(["profile_baseline", "time_decay", "hybrid_semantic", "implicit_mf"])].copy()
    pivot = subset.pivot(index="model", columns="segment", values="ndcg_at_k").sort_index()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("NDCG@5 по сегментам для основных моделей")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save(fig, OUTPUT_DIR / "segment_metrics.png")


def generate_project_tree() -> None:
    tree_text = build_project_tree(ROOT)
    render_text_card(
        OUTPUT_DIR / "project_tree.png",
        "Структура проекта",
        tree_text,
        subtitle="снимок рабочего каталога",
        mono=True,
        width=70,
        max_lines=42,
    )


def generate_notebook_preview() -> None:
    nb = json.loads((ROOT / "notebooks" / "01_eda_transactions.ipynb").read_text(encoding="utf-8"))
    markdown_cells = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    parts = [
        "# Обзор ноутбука",
        "".join(markdown_cells[0].get("source", [])) if markdown_cells else "",
        "## Пример кодовой ячейки",
        "".join(code_cells[0].get("source", [])) if code_cells else "",
    ]
    render_text_card(
        OUTPUT_DIR / "notebook_preview.png",
        "Фрагмент EDA-ноутбука",
        "\n\n".join(parts),
        subtitle="01_eda_transactions.ipynb",
        mono=False,
        width=92,
        max_lines=34,
    )


def generate_model_report_preview() -> None:
    body = extract_text(ROOT / "reports" / "analysis_summary_report.md", max_lines=36)
    render_text_card(
        OUTPUT_DIR / "model_report_preview.png",
        "Фрагмент отчета по сравнению моделей",
        body,
        subtitle="analysis_summary_report.md",
        mono=True,
        width=96,
        max_lines=34,
    )


def generate_model_module_preview() -> None:
    body = extract_text(ROOT / "src" / "models" / "time_decay_recommender.py", max_lines=42)
    render_text_card(
        OUTPUT_DIR / "model_module_preview.png",
        "Фрагмент модуля модели",
        body,
        subtitle="src/models/time_decay_recommender.py",
        mono=True,
        width=96,
        max_lines=38,
    )


def generate_run_all_script_preview() -> None:
    body = extract_text(ROOT / "scripts" / "run_all.ps1", max_lines=42)
    render_text_card(
        OUTPUT_DIR / "run_all_script_preview.png",
        "Фрагмент сценария полного прогона",
        body,
        subtitle="scripts/run_all.ps1",
        mono=True,
        width=96,
        max_lines=38,
    )


def generate_api_response_fallback() -> None:
    payload = json.loads((ROOT / "reports" / "service_demo_output.json").read_text(encoding="utf-8"))
    example = {"user_id": "U00001", "recommendations": payload.get("U00001", [])[:2]}
    body = json.dumps(example, ensure_ascii=False, indent=2)
    render_text_card(
        OUTPUT_DIR / "api_response_example.png",
        "Пример ответа API",
        body,
        subtitle="GET /recommend/U00001?top_k=5",
        mono=True,
        width=88,
        max_lines=34,
    )


def generate_swagger_fallback() -> None:
    body = "\n".join(
        [
            "FastAPI-приложение: сервис персонализированных банковских предложений",
            "",
            "GET  /                     Корневой endpoint",
            "GET  /health               Проверка состояния сервиса",
            "GET  /recommend/{user_id}  Персонализированные рекомендации top-K",
            "",
            "OpenAPI title: Сервис персонализации банковских предложений",
            "Версия: 0.1.0",
            "",
            "Живой Swagger UI можно получить отдельно при запущенном локальном сервисе.",
        ]
    )
    render_text_card(
        OUTPUT_DIR / "swagger_ui.png",
        "Предпросмотр Swagger UI",
        body,
        subtitle="локальная документация API",
        mono=False,
        width=88,
        max_lines=24,
    )


def main() -> None:
    generate_pipeline_architecture()
    generate_dataset_structure()
    generate_profile_scheme()
    generate_synthetic_model_metrics()
    generate_multiseed_summary()
    generate_segment_distribution()
    generate_segment_spend_summary()
    generate_time_decay_tuning()
    generate_bootstrap_ci_chart()
    generate_segment_metrics()
    generate_project_tree()
    generate_notebook_preview()
    generate_model_report_preview()
    generate_model_module_preview()
    generate_run_all_script_preview()
    if not (OUTPUT_DIR / "swagger_ui.png").exists():
        generate_swagger_fallback()
    generate_api_response_fallback()
    print(f"[ok] thesis figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
