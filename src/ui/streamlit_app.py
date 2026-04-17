from __future__ import annotations

import random
from html import escape

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from src.service.backend import RecommendationBackend
from src.service.localization import translate_category, translate_model, translate_segment


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');
        :root{--bg:#f4efe6;--paper:rgba(255,251,245,.9);--ink:#142620;--muted:#667065;--line:rgba(20,38,32,.1);--teal:#123f3b;--teal2:#0f6c63;--copper:#c48d36;--shadow:0 22px 52px rgba(24,38,34,.12)}
        @keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
        .stApp{background:radial-gradient(circle at top left, rgba(196,141,54,.18), transparent 24%),radial-gradient(circle at 85% 10%, rgba(15,108,99,.16), transparent 22%),linear-gradient(180deg,#f5efe5 0%,#efe6d8 48%,#f7f2ea 100%);color:var(--ink);font-family:"Manrope","Segoe UI",sans-serif}
        h1,h2,h3,.display-title,.section-title,.offer-name,.big-number{font-family:"Fraunces",Georgia,serif}
        header[data-testid="stHeader"]{background:transparent}
        [data-testid="stToolbar"],[data-testid="stStatusWidget"],#MainMenu,.stAppDeployButton{display:none!important}
        .block-container{max-width:1360px;padding-top:1rem;padding-bottom:3rem;padding-left:2rem;padding-right:2rem}
        [data-testid="stSidebar"]{background:linear-gradient(180deg, rgba(255,250,244,.94), rgba(241,236,224,.96));border-right:1px solid rgba(20,38,32,.08)}
        [data-testid="stSidebar"] .block-container{padding-top:1.2rem;padding-left:1rem;padding-right:1rem}
        .sidebar-brand,.hero-shell,.signal-strip,.surface-panel,.offer-card{animation:fadeUp .55s ease both}
        .sidebar-brand{border-radius:28px;padding:1.1rem 1rem;background:linear-gradient(135deg, rgba(18,63,59,.98), rgba(31,87,82,.94));color:#fdf7ef;box-shadow:0 20px 38px rgba(20,38,34,.16);margin-bottom:1rem}
        .sidebar-brand .eyebrow,.hero-kicker,.stat-label{text-transform:uppercase;letter-spacing:.14em;font-size:.7rem;opacity:.76}
        .sidebar-brand h2{margin:.35rem 0 0 0;color:#fdf7ef;font-size:1.34rem}
        .sidebar-brand p{margin:.5rem 0 0 0;color:rgba(253,247,239,.88);font-size:.88rem;line-height:1.5}
        div[data-baseweb="select"]>div,[data-baseweb="base-input"]>div{border-radius:18px!important;border-color:rgba(20,38,32,.1)!important;background:rgba(255,251,245,.82)!important;box-shadow:none!important}
        .stButton>button{width:100%;min-height:3rem;border-radius:999px;border:1px solid rgba(20,38,32,.12);background:linear-gradient(180deg, rgba(255,255,255,.92), rgba(247,241,232,.92));color:var(--ink);font-weight:600}
        [data-testid="stSlider"] [role="slider"]{background:var(--copper);border-color:var(--copper)}
        .hero-shell{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.75fr);gap:1rem;border-radius:36px;padding:2rem;margin-bottom:1rem;color:#faf6f0;background:radial-gradient(circle at top right, rgba(196,141,54,.24), transparent 28%),linear-gradient(135deg,#10231f 0%,#123f3b 44%,#1d635c 100%);box-shadow:0 30px 70px rgba(20,38,34,.22)}
        .display-title{margin:.45rem 0 0 0;font-size:clamp(2.7rem,4vw,4.6rem);line-height:.96;letter-spacing:-.04em;max-width:14ch}
        .hero-subtitle{margin-top:.95rem;max-width:42rem;color:rgba(250,246,240,.86);font-size:1rem;line-height:1.6}
        .hero-chip-row,.tag-row{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}
        .hero-chip,.tag-chip,.pill{display:inline-flex;align-items:center;border-radius:999px;padding:.42rem .72rem;border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.1);font-size:.8rem;line-height:1}
        .hero-rail{border-radius:28px;padding:1.05rem;background:linear-gradient(180deg, rgba(255,255,255,.12), rgba(255,255,255,.05));border:1px solid rgba(255,255,255,.12)}
        .hero-rail h3{margin:0;font-size:1.08rem;color:#faf6f0}.hero-rail p{margin:.35rem 0 0 0;font-size:.9rem;line-height:1.5;color:rgba(250,246,240,.8)}
        .hero-grid,.strip-grid,.portrait-grid{display:grid;gap:.75rem}
        .hero-grid{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:1rem}
        .strip-grid{grid-template-columns:repeat(5,minmax(0,1fr))}
        .portrait-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
        .mini-card,.strip-card,.stat-card{border-radius:22px;padding:.9rem;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.08)}
        .mini-card .big-number,.strip-card .big-number,.stat-card .big-number{margin-top:.25rem;font-size:1.2rem;line-height:1.05}
        .signal-strip{border-radius:28px;padding:.35rem;background:rgba(255,251,245,.76);border:1px solid var(--line);box-shadow:0 18px 36px rgba(36,46,42,.08);margin-bottom:1.3rem}
        .strip-card{background:transparent;border:none;border-right:1px solid var(--line);min-height:116px;display:flex;flex-direction:column;justify-content:space-between}
        .strip-grid .strip-card:last-child{border-right:none}
        .surface-panel{border-radius:28px;padding:1.08rem;background:var(--paper);border:1px solid var(--line);box-shadow:0 18px 34px rgba(34,45,40,.08)}
        .section-title{font-size:1.7rem;line-height:1;letter-spacing:-.02em}.section-copy{margin-top:.35rem;color:var(--muted);font-size:.94rem;line-height:1.55}
        .stat-card{background:linear-gradient(180deg, rgba(255,255,255,.86), rgba(245,239,228,.8));border:1px solid rgba(20,38,32,.06)}
        .offer-card{border-radius:28px;padding:1.16rem;margin-bottom:.9rem;background:linear-gradient(180deg, rgba(255,255,255,.96), rgba(247,242,234,.92));border:1px solid rgba(20,38,32,.08);box-shadow:0 20px 42px rgba(33,43,40,.08)}
        .offer-top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}.offer-rank{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:#6e796d}
        .offer-name{margin:.28rem 0 0 0;font-size:1.42rem;line-height:1.04;max-width:22ch}.offer-type{border-radius:999px;padding:.38rem .74rem;background:rgba(18,63,59,.08);border:1px solid rgba(18,63,59,.1);color:var(--teal2);font-size:.8rem}
        .offer-score{display:inline-flex;margin-top:.55rem;border-radius:999px;padding:.34rem .68rem;background:rgba(196,141,54,.12);color:#805915;font-size:.78rem;font-weight:700}
        .offer-description,.offer-reason,.mini-note{line-height:1.58}.offer-description{margin-top:.75rem;color:#49574d;font-size:.94rem}.offer-reason{margin-top:.82rem;color:#20332b;font-size:.94rem}.mini-note{margin-top:.62rem;color:#5d695e;font-size:.85rem}
        .tag-chip{background:rgba(15,108,99,.08);border:1px solid rgba(15,108,99,.1);color:#25544d}
        .score-track{position:relative;margin-top:.92rem;height:9px;border-radius:999px;background:rgba(20,38,32,.08);overflow:hidden}.score-fill{position:absolute;inset:0 auto 0 0;border-radius:999px;background:linear-gradient(90deg, var(--teal2), var(--copper))}
        .stTabs [data-baseweb="tab-list"]{gap:.45rem;padding:.35rem;border-radius:999px;background:rgba(255,251,245,.78);border:1px solid var(--line)}
        .stTabs [data-baseweb="tab"]{border-radius:999px!important;color:#556458!important;font-weight:600!important;padding-left:1rem!important;padding-right:1rem!important;min-height:2.7rem!important}
        .stTabs [data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg, var(--teal), var(--teal2))!important;color:#faf6f0!important;box-shadow:0 12px 26px rgba(18,63,59,.18)}
        [data-testid="stDataFrame"],[data-testid="stVegaLiteChart"],[data-testid="stImage"]{background:rgba(255,252,247,.84);border:1px solid var(--line);border-radius:24px;overflow:hidden;box-shadow:0 14px 28px rgba(37,46,42,.07)}
        [data-testid="stVegaLiteChart"],[data-testid="stImage"]{padding:.6rem .65rem .15rem .65rem}
        .timeline{display:flex;flex-direction:column;gap:.75rem}.timeline-item{display:grid;grid-template-columns:82px 1fr;gap:.8rem;padding-top:.7rem;border-top:1px solid rgba(20,38,32,.08)}.timeline-item:first-child{padding-top:0;border-top:0}
        .timeline-date{font-size:.78rem;color:#718072;text-transform:uppercase;letter-spacing:.08em}.timeline-offer{color:var(--ink);font-weight:700;line-height:1.35}.timeline-meta{margin-top:.18rem;color:#5e695f;font-size:.86rem}
        .leaderboard{display:flex;flex-direction:column;gap:.55rem}.leader-row{display:grid;grid-template-columns:52px 1.15fr .9fr;gap:.8rem;align-items:center;padding:.8rem .9rem;border-radius:20px;background:rgba(255,255,255,.78);border:1px solid rgba(20,38,32,.06)}.leader-row.best{background:linear-gradient(135deg, rgba(18,63,59,.94), rgba(30,90,84,.92));box-shadow:0 18px 34px rgba(20,36,33,.14)}
        .leader-rank{color:#7a8578;font-weight:700;font-size:.9rem}.leader-model{color:var(--ink);font-weight:700;line-height:1.35}.leader-metrics{text-align:right;color:#536055;font-size:.86rem;line-height:1.45}.leader-row.best .leader-rank,.leader-row.best .leader-model,.leader-row.best .leader-metrics{color:#faf6f0}
        .footer-note{margin-top:1rem;border-radius:28px;padding:1rem 1.1rem;background:rgba(255,251,245,.72);border:1px solid var(--line);color:#586459;font-size:.9rem;line-height:1.6}
        @media (max-width:1120px){.hero-shell{grid-template-columns:1fr}.strip-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.portrait-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.strip-card{border-right:none;border-bottom:1px solid var(--line)}}
        @media (max-width:760px){.block-container{padding-left:1rem;padding-right:1rem}.display-title{font-size:2.4rem}.hero-grid,.strip-grid,.portrait-grid{grid-template-columns:1fr}.offer-top{flex-direction:column;align-items:flex-start}.timeline-item,.leader-row{grid-template-columns:1fr}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_backend() -> RecommendationBackend:
    backend = RecommendationBackend()
    backend.load()
    return backend


def safe(value: object) -> str:
    return escape("" if value is None else str(value))


def format_currency(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def render_section(title: str, copy: str) -> None:
    st.markdown(
        f'<div style="margin:0.25rem 0 0.85rem 0"><div class="section-title">{safe(title)}</div><div class="section-copy">{safe(copy)}</div></div>',
        unsafe_allow_html=True,
    )


def render_chart(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_offer_card(row: pd.Series) -> None:
    tags = [tag.strip() for tag in str(row["target_categories_label"]).split(",") if tag.strip()]
    tag_html = "".join(f'<span class="tag-chip">{safe(tag)}</span>' for tag in tags)
    fit_score = float(row["fit_score_pct"])
    matched = str(row.get("matched_categories", "")).strip()
    matched_line = f"Ключевые совпадения: {matched}" if matched else "Ключевые совпадения: широкое соответствие профилю."
    st.markdown(
        f"""
        <article class="offer-card">
          <div class="offer-top">
            <div>
              <div class="offer-rank">Рекомендация #{int(row["rank"])}</div>
              <div class="offer-name">{safe(row["offer_name_label"])}</div>
            </div>
            <div class="offer-type">{safe(row["product_type_label"])}</div>
          </div>
          <div class="offer-score">Score {fit_score:.1f}%</div>
          <div class="offer-description">{safe(row["description_label"])}</div>
          <div class="tag-row">{tag_html}</div>
          <div class="offer-reason">{safe(row["reason"])}</div>
          <div class="mini-note">{safe(matched_line)}</div>
          <div class="score-track"><div class="score-fill" style="width:{fit_score:.1f}%"></div></div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def build_category_chart(category_mix: pd.DataFrame) -> plt.Figure:
    chart_df = category_mix.head(8).copy()
    chart_df["share_pct"] = chart_df["spend_share"] * 100
    chart_df = chart_df.sort_values("share_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    max_value = float(chart_df["share_pct"].max()) if not chart_df.empty else 1.0
    colors = ["#c48d36" if value == max_value else "#17665f" for value in chart_df["share_pct"]]
    bars = ax.barh(chart_df["label"], chart_df["share_pct"], color=colors, height=0.55)

    ax.set_xlim(0, max_value * 1.18 if max_value else 1)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", colors="#445348", labelsize=11, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for bar, value in zip(bars, chart_df["share_pct"], strict=False):
        ax.text(
            bar.get_width() + max_value * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="#20332b",
            fontweight="semibold",
        )

    plt.tight_layout()
    return fig


def build_monthly_chart(monthly_spend: pd.DataFrame) -> plt.Figure:
    chart_df = monthly_spend.copy().sort_values("month")
    chart_df["month"] = pd.to_datetime(chart_df["month"])

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.fill_between(chart_df["month"], chart_df["amount"], color="#d9b878", alpha=0.18)
    ax.plot(chart_df["month"], chart_df["amount"], color="#123f3b", linewidth=2.6)
    ax.scatter(chart_df["month"], chart_df["amount"], s=34, color="#c48d36", edgecolors="#faf6f0", linewidths=1.2, zorder=3)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{int(value):,}".replace(",", " ")))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", colors="#516156", labelsize=10, rotation=0, length=0)
    ax.tick_params(axis="y", colors="#516156", labelsize=10, length=0)
    ax.grid(axis="y", color="#d9d1c4", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def build_model_chart(df: pd.DataFrame) -> plt.Figure:
    ordered = df.sort_values("ndcg_at_k", ascending=True).copy()
    ordered["model_label"] = ordered["model"].map(translate_model)
    best_value = float(ordered["ndcg_at_k"].max()) if not ordered.empty else 0.0

    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = ["#c48d36" if value == best_value else "#17665f" for value in ordered["ndcg_at_k"]]
    bars = ax.barh(ordered["model_label"], ordered["ndcg_at_k"], color=colors, height=0.58)

    ax.set_xlim(0, best_value * 1.18 if best_value else 1)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", colors="#516156", labelsize=10, length=0)
    ax.tick_params(axis="y", colors="#445348", labelsize=11, length=0)
    ax.grid(axis="x", color="#d9d1c4", linewidth=0.8, alpha=0.8)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for bar, value in zip(bars, ordered["ndcg_at_k"], strict=False):
        ax.text(
            bar.get_width() + max(best_value * 0.02, 0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#20332b",
            fontweight="semibold",
        )

    plt.tight_layout()
    return fig


def render_sidebar(backend: RecommendationBackend) -> tuple[str | None, str, int]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="eyebrow">Elena thesis demo</div>
          <h2>Offer Intelligence</h2>
          <p>Демонстрационная витрина для ВКР: профиль клиента, explainable-рекомендации и benchmark моделей.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    segments = ["Все сегменты"] + [translate_segment(segment) for segment in backend.get_segments()]
    segment_lookup = {translate_segment(segment): segment for segment in backend.get_segments()}
    selected_segment_label = st.sidebar.selectbox("Сегмент клиента", segments)
    selected_segment = segment_lookup.get(selected_segment_label)

    user_options = backend.get_user_options(selected_segment)
    if not user_options:
        st.sidebar.error("Для выбранного сегмента нет пользователей.")
        st.stop()

    if "selected_user" not in st.session_state or st.session_state["selected_user"] not in user_options:
        st.session_state["selected_user"] = user_options[0]

    if st.sidebar.button("Случайный клиент", use_container_width=True):
        st.session_state["selected_user"] = random.choice(user_options)

    selected_user = st.sidebar.selectbox(
        "Профиль клиента",
        user_options,
        key="selected_user",
        format_func=backend.get_user_label,
    )
    top_k = st.sidebar.slider("Размер рекомендательной выдачи", min_value=3, max_value=10, value=5)
    st.sidebar.caption("Интерфейс работает на синтетическом банковском датасете и показывает активную serving-модель Time-Decay.")
    return selected_segment, selected_user, top_k


def render_hero(summary: dict[str, object], snapshot: dict[str, object], top_k: int) -> None:
    last_tx = snapshot["last_tx_date"]
    last_tx_display = last_tx.strftime("%d.%m.%Y") if isinstance(last_tx, pd.Timestamp) else "н/д"
    prefs = snapshot["top_preference_categories"]
    pref_html = "".join(
        f"<span class='hero-chip'>{safe(translate_category(str(row.category)))}</span>"
        for row in prefs.head(4).itertuples(index=False)
    )
    chips = [
        f"<span class='hero-chip'>Сегмент: {safe(snapshot['segment_label'])}</span>",
        f"<span class='hero-chip'>Клиент: {safe(snapshot['user_id'])}</span>",
        f"<span class='hero-chip'>Топ-{int(top_k)}</span>",
    ]
    best_ndcg = summary["best_ndcg_at_k"]
    hero_stats = [
        ("Активная модель", "Time-Decay"),
        ("Лидер бенчмарка", f"{safe(summary['best_model_label'])} · {best_ndcg:.4f}" if isinstance(best_ndcg, float) else safe(summary["best_model_label"])),
        ("Последняя активность", last_tx_display),
        ("Средний чек", f"{format_currency(float(snapshot['avg_ticket']))} ₽"),
    ]
    stats_html = "".join(
        f"<div class='mini-card'><div class='stat-label'>{safe(label)}</div><div class='big-number'>{safe(value)}</div></div>"
        for label, value in hero_stats
    )

    st.markdown(
        f"""
        <section class="hero-shell">
          <div>
            <div class="hero-kicker">Bank Offer Intelligence</div>
            <div class="display-title">Персональные банковские предложения, которые можно объяснить.</div>
            <div class="hero-subtitle">Профиль клиента, динамика трат и explainable top-K выдача собраны в один аккуратный демонстрационный экран вместо “сырого Streamlit”-прототипа.</div>
            <div class="hero-chip-row">{''.join(chips)}</div>
          </div>
          <aside class="hero-rail">
            <h3>Текущий сигнал модели</h3>
            <p>Показываем не только результат, но и контекст: кто клиент, насколько свежа активность и какие темы формируют выдачу.</p>
            <div class="hero-grid">{stats_html}</div>
            <div class="hero-chip-row">{pref_html}</div>
          </aside>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_signal_strip(summary: dict[str, object]) -> None:
    positive_rate = summary["positive_rate"]
    best_ndcg = summary["best_ndcg_at_k"]
    items = [
        ("Клиенты", str(summary["n_users"]), "Синтетические профили, доступные в витрине."),
        ("Транзакции", format_currency(float(summary["n_transactions"])), "Общий массив событий, формирующий поведенческий контур."),
        ("Каталог", str(summary["n_offers"]), "Количество офферов, которые модель ранжирует."),
        ("Лидер бенчмарка", str(summary["best_model_label"]), f"NDCG@5 = {best_ndcg:.4f}" if isinstance(best_ndcg, float) else "Метрика недоступна"),
        ("Позитивный отклик", f"{positive_rate:.1%}" if isinstance(positive_rate, float) else "н/д", "Симулированная доля успешных откликов."),
    ]
    html = "".join(
        f"<div class='strip-card'><div><div class='stat-label'>{safe(label)}</div><div class='big-number'>{safe(value)}</div></div><div class='mini-note'>{safe(note)}</div></div>"
        for label, value, note in items
    )
    st.markdown(f"<section class='signal-strip'><div class='strip-grid'>{html}</div></section>", unsafe_allow_html=True)


def render_profile_panel(snapshot: dict[str, object]) -> None:
    last_tx = snapshot["last_tx_date"]
    last_tx_display = last_tx.strftime("%d.%m.%Y") if isinstance(last_tx, pd.Timestamp) else "н/д"
    stats = [
        ("Сегмент", str(snapshot["segment_label"]), "Поведенческий кластер клиента."),
        ("Транзакции", str(int(snapshot["tx_count"])), "Количество операций в истории."),
        ("Суммарные траты", f"{format_currency(float(snapshot['total_spend']))} ₽", "Интегральный объем активности."),
        ("Средний чек", f"{format_currency(float(snapshot['avg_ticket']))} ₽", "Средняя сумма одной операции."),
        ("Последняя активность", last_tx_display, "Насколько свеж сигнал для модели."),
    ]
    cards = "".join(
        f"<div class='stat-card'><div class='stat-label'>{safe(label)}</div><div class='big-number'>{safe(value)}</div><div class='mini-note'>{safe(note)}</div></div>"
        for label, value, note in stats
    )
    st.markdown(
        f"<div class='surface-panel'><div class='section-title'>Портрет клиента</div><div class='section-copy'>Ключевые сигналы, которыми удобно объяснять персонализацию на демо и защите.</div><div class='portrait-grid' style='margin-top:.9rem'>{cards}</div></div>",
        unsafe_allow_html=True,
    )


def render_preference_panel(snapshot: dict[str, object]) -> None:
    prefs = snapshot["top_preference_categories"]
    if prefs.empty:
        st.info("Профиль предпочтений для этого клиента недоступен.")
        return
    max_value = float(prefs["value"].max()) if not prefs.empty else 1.0
    rows = []
    for row in prefs.head(6).itertuples(index=False):
        width = float(row.value) / max_value * 100 if max_value else 0.0
        rows.append(
            f'<div style="margin-top:.75rem"><div style="display:flex;justify-content:space-between;gap:.75rem;color:#142620;font-size:.92rem"><span>{safe(translate_category(str(row.category)))}</span><strong>{float(row.value):.2f}</strong></div><div class="score-track" style="margin-top:.4rem;height:10px"><div class="score-fill" style="width:{width:.1f}%"></div></div></div>'
        )
    st.markdown(
        f"<div class='surface-panel'><div class='section-title'>Сигнатура интересов</div><div class='section-copy'>Главные темы трат, которые модель затем использует для объяснения офферов.</div>{''.join(rows)}<div class='mini-note'>Показаны относительные веса категорий, а не абсолютные банковские KPI.</div></div>",
        unsafe_allow_html=True,
    )


def render_timeline(accepted_df: pd.DataFrame) -> None:
    if accepted_df.empty:
        st.info("У клиента нет позитивных откликов в наблюдаемой истории.")
        return
    accepted = accepted_df.copy()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"]).dt.strftime("%d.%m.%Y")
    items = "".join(
        f"<div class='timeline-item'><div class='timeline-date'>{safe(row.timestamp)}</div><div><div class='timeline-offer'>{safe(row.offer_name_label)}</div><div class='timeline-meta'>{safe(row.product_type_label)}</div></div></div>"
        for row in accepted.itertuples(index=False)
    )
    st.markdown(
        f"<div class='surface-panel'><div class='section-title'>История позитивных откликов</div><div class='section-copy'>Контекст для разговора о том, насколько новая выдача отличается от уже принятых офферов.</div><div class='timeline' style='margin-top:.8rem'>{items}</div></div>",
        unsafe_allow_html=True,
    )


def render_recommendation_brief(snapshot: dict[str, object], recommendations_df: pd.DataFrame) -> None:
    lead_offer = recommendations_df.iloc[0]["offer_name_label"] if not recommendations_df.empty else "н/д"
    mean_fit = recommendations_df["fit_score_pct"].mean() if not recommendations_df.empty else 0.0
    product_mix = ", ".join(recommendations_df["product_type_label"].astype(str).drop_duplicates().head(3).tolist())
    dominant = ", ".join(
        translate_category(str(row.category)) for row in snapshot["top_preference_categories"].head(3).itertuples(index=False)
    )
    st.markdown(
        f"""
        <div class='surface-panel'>
          <div class='section-title'>Почему выдача выглядит так</div>
          <div class='section-copy'>Короткий narrative-блок для демонстрации: на что смотреть в первую очередь и как объяснять логику модели.</div>
          <div class='portrait-grid' style='grid-template-columns:1fr; margin-top:.9rem'>
            <div class='stat-card'><div class='stat-label'>Лидирующий оффер</div><div class='big-number'>{safe(lead_offer)}</div><div class='mini-note'>Первый кандидат, который удобнее всего разбирать на защите.</div></div>
            <div class='stat-card'><div class='stat-label'>Средний score</div><div class='big-number'>{mean_fit:.1f}%</div><div class='mini-note'>Средняя сила совпадения среди офферов в текущем top-K.</div></div>
            <div class='stat-card'><div class='stat-label'>Доминирующие темы</div><div class='mini-note'>{safe(dominant or 'н/д')}</div><div class='mini-note'><strong>Типы продуктов:</strong> {safe(product_mix or 'н/д')}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_leaderboard(df: pd.DataFrame, title: str, subtitle: str) -> None:
    if df.empty:
        st.info("Файл бенчмарка не найден.")
        return
    ordered = df.sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)
    rows = []
    for idx, row in enumerate(ordered.itertuples(index=False), start=1):
        rows.append(
            f"<div class='leader-row {'best' if idx == 1 else ''}'><div class='leader-rank'>{idx:02d}</div><div class='leader-model'>{safe(translate_model(str(row.model)))}</div><div class='leader-metrics'><div>NDCG@K {float(row.ndcg_at_k):.4f}</div><div>MAP@K {float(row.map_at_k):.4f}</div></div></div>"
        )
    st.markdown(
        f"<div class='surface-panel'><div class='section-title'>{safe(title)}</div><div class='section-copy'>{safe(subtitle)}</div><div class='leaderboard' style='margin-top:.9rem'>{''.join(rows)}</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Витрина персональных банковских предложений", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    backend = get_backend()
    _, user_id, top_k = render_sidebar(backend)

    summary = backend.get_overall_summary()
    snapshot = backend.get_user_snapshot(user_id)
    recommendations_df = backend.recommend(user_id, top_k)
    segment_benchmark_df = backend.get_segment_benchmark(str(snapshot["segment"]))
    overall_benchmark_df = backend.overall_metrics_df if backend.overall_metrics_df is not None else pd.DataFrame()

    render_hero(summary, snapshot, top_k)
    render_signal_strip(summary)

    overview_left, overview_right = st.columns([1.08, 0.92], gap="large")
    with overview_left:
        render_profile_panel(snapshot)
    with overview_right:
        render_preference_panel(snapshot)

    tabs = st.tabs(["Рекомендации", "Активность клиента", "Сравнение моделей"])

    with tabs[0]:
        render_section("Персональные предложения", "Основная витрина: офферы, которые модель считает наиболее релевантными для текущего клиента.")
        rec_left, rec_right = st.columns([1.28, 0.72], gap="large")
        with rec_left:
            for _, rec_row in recommendations_df.iterrows():
                render_offer_card(rec_row)
        with rec_right:
            render_recommendation_brief(snapshot, recommendations_df)
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            render_timeline(snapshot["accepted_offers"])

    with tabs[1]:
        render_section("Транзакционная активность", "Откуда модель берет сигналы: структура трат, динамика и недавние события клиента.")
        category_mix = snapshot["category_mix"].copy()
        if not category_mix.empty:
            category_mix["label"] = category_mix["category_label"]
        chart_left, chart_right = st.columns([1.02, 0.98], gap="large")
        with chart_left:
            if category_mix.empty:
                st.info("Для этого клиента нет транзакционных данных.")
            else:
                render_chart(build_category_chart(category_mix))
        with chart_right:
            monthly_spend = snapshot["monthly_spend"].copy()
            if monthly_spend.empty:
                st.info("Помесячная динамика недоступна.")
            else:
                render_chart(build_monthly_chart(monthly_spend))

        table_left, table_right = st.columns([1.1, 0.9], gap="large")
        with table_left:
            render_section("Последние транзакции", "События, которые помогают объяснять краткосрочную составляющую Time-Decay.")
            recent = snapshot["recent_transactions"].copy()
            if recent.empty:
                st.info("Последние транзакции недоступны.")
            else:
                recent["timestamp"] = pd.to_datetime(recent["timestamp"]).dt.strftime("%d.%m.%Y %H:%M")
                recent["amount"] = recent["amount"].map(lambda value: f"{value:,.0f}".replace(",", " "))
                recent = recent.rename(columns={"timestamp": "Дата", "category_label": "Категория", "amount": "Сумма, ₽", "channel_label": "Канал"})[["Дата", "Категория", "Сумма, ₽", "Канал"]]
                st.dataframe(recent, use_container_width=True, hide_index=True)
        with table_right:
            render_section("Разбивка по категориям", "Сводная таблица по основным направлениям расходов клиента.")
            if category_mix.empty:
                st.info("Разбивка по категориям недоступна.")
            else:
                breakdown = category_mix[["label", "total_amount", "tx_count"]].copy()
                breakdown["total_amount"] = breakdown["total_amount"].map(lambda value: f"{value:,.0f}".replace(",", " "))
                breakdown = breakdown.rename(columns={"label": "Категория", "total_amount": "Сумма, ₽", "tx_count": "Транзакции"})
                st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with tabs[2]:
        render_section("Сравнение моделей", "Общий рейтинг и срез по текущему сегменту рядом: так быстрее видно, где лидер стабилен, а где поведение меняется.")
        bench_left, bench_right = st.columns([1.0, 1.0], gap="large")
        with bench_left:
            render_model_leaderboard(overall_benchmark_df, "Общий рейтинг", "Synthetic benchmark по всем пользователям.")
            if not overall_benchmark_df.empty:
                render_chart(build_model_chart(overall_benchmark_df))
        with bench_right:
            render_model_leaderboard(segment_benchmark_df, f'Сегмент: {snapshot["segment_label"]}', "Срез качества по выбранному поведенческому сегменту.")
            if not segment_benchmark_df.empty:
                render_chart(build_model_chart(segment_benchmark_df))
        st.markdown(
            "<div class='footer-note'><strong>Как читать benchmark.</strong> Общий рейтинг показывает усредненное качество по synthetic датасету, а срез по сегменту помогает объяснить, почему одна и та же модель может выглядеть сильнее или слабее в зависимости от типа клиента и структуры его расходов.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
