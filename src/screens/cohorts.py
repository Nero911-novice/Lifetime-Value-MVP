from __future__ import annotations

import pandas as pd
import streamlit as st

from ..metrics import (
    build_cohort_user_base,
    get_cohort_summary,
    get_cohort_maturity_table,
    build_retention_matrix,
    build_cumulative_ltv_matrix,
    build_cumulative_margin_matrix,
    build_cancellation_matrix,
    build_promo_share_matrix,
    build_rides_per_user_matrix,
    build_cohort_size_matrix,
    get_selected_cohort_profile,
    get_selected_cohort_curves,
    compare_cohort_to_baseline,
    generate_cohort_diagnostics,
)
from ..ui import render_screen_help, render_item_help, render_methodology_footer, format_currency, format_number, format_percent


def render_cohort_header() -> None:
    st.header("Когорты")
    st.caption("Диагностический экран качества когорт: размер, удержание, монетизация, маржинальность, отмены и промо-зависимость.")
    render_screen_help("cohorts")


def render_cohort_filters_info(cohort_summary: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    if cohort_summary.empty:
        st.warning("После фильтрации нет активированных когорт. Измените глобальные фильтры.")
        return cohort_summary, 0, "Retention"
    options = sorted(cohort_summary["cohort_month"].unique().tolist())
    period = st.select_slider("Диапазон месяцев когорт", options=options, value=(options[0], options[-1]))
    min_maturity = st.slider("Минимальная зрелость когорты, месяцев", min_value=0, max_value=18, value=1, step=1)
    mode = st.selectbox(
        "Режим основной матрицы",
        options=[
            "Retention",
            "Накопленный LTV",
            "Накопленная маржа",
            "Доля отмен",
            "Доля промо-поездок",
            "Поездки на пользователя",
            "Размер когорты",
        ],
    )
    filtered = cohort_summary.loc[
        cohort_summary["cohort_month"].between(period[0], period[1]) & (cohort_summary["maturity_months"] >= min_maturity)
    ].copy()
    return filtered, min_maturity, mode


def render_cohort_kpis(cohort_summary: pd.DataFrame) -> None:
    if cohort_summary.empty:
        return
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Число когорт", format_number(len(cohort_summary), 0))
    k2.metric("Суммарный размер", format_number(cohort_summary["cohort_size"].sum(), 0))
    k3.metric("Средняя зрелость", f"{cohort_summary['maturity_months'].mean():.1f} мес.")
    k4.metric("Средний retention M1", format_percent(cohort_summary["retention_m1"].mean(skipna=True), 1))
    k5.metric("Средний retention M3", format_percent(cohort_summary["retention_m3"].mean(skipna=True), 1))
    k6, k7, k8, k9 = st.columns(4)
    k6.metric("Средний LTV 90д", format_currency(cohort_summary["avg_ltv_90d"].mean(), 0))
    k7.metric("Средний LTV 180д", format_currency(cohort_summary["avg_ltv_180d"].mean(), 0))
    k8.metric("Средняя доля отмен", format_percent(cohort_summary["avg_cancellation_rate"].mean(), 1))
    k9.metric("Средняя доля промо", format_percent(cohort_summary["avg_promo_trip_share"].mean(), 1))


def _render_matrix(matrix: pd.DataFrame, mode: str) -> None:
    if matrix.empty:
        st.info("Нет данных для отображения матрицы в текущем фильтре.")
        return
    styled = matrix.style
    if mode in {"Retention", "Доля отмен", "Доля промо-поездок"}:
        styled = styled.format("{:.1%}").background_gradient(cmap="Blues")
    elif mode in {"Накопленный LTV", "Накопленная маржа"}:
        styled = styled.format("{:,.0f} ₽").background_gradient(cmap="Greens")
    elif mode == "Поездки на пользователя":
        styled = styled.format("{:,.2f}").background_gradient(cmap="Purples")
    else:
        styled = styled.format("{:,.0f}")
    st.dataframe(styled, use_container_width=True)


def render_cohort_main_matrix(mode: str, matrices: dict[str, pd.DataFrame]) -> None:
    st.subheader("Основная когортная матрица")
    mapping = {
        "Retention": ("retention", "cohort_retention_help"),
        "Накопленный LTV": ("ltv", "cohort_ltv_help"),
        "Накопленная маржа": ("margin", "cohort_margin_help"),
        "Доля отмен": ("cancellation", "cohort_cancellation_help"),
        "Доля промо-поездок": ("promo_share", "cohort_promo_share_help"),
        "Поездки на пользователя": ("rides", "cohort_rides_per_user_help"),
        "Размер когорты": ("size", "cohort_size_help"),
    }
    key, help_key = mapping[mode]
    _render_matrix(matrices[key], mode)
    render_item_help(help_key, "Пояснение к режиму матрицы")


def render_cohort_maturity_table(maturity_table: pd.DataFrame) -> None:
    st.subheader("Таблица зрелости когорт")
    if maturity_table.empty:
        st.info("Нет зрелых когорт в текущем фильтре.")
        return
    st.dataframe(maturity_table.rename(columns={"cohort_month": "Когорта", "cohort_size": "Размер", "maturity_months": "Зрелость, месяцев"}), use_container_width=True)
    render_item_help("cohort_maturity_help", "Пояснение к зрелости")


def render_selected_cohort_compare(cohort_summary: pd.DataFrame, selected_cohort: str) -> tuple[dict, dict]:
    st.subheader("Сравнение выбранной когорты с эталоном (медиана)")
    compare_table = compare_cohort_to_baseline(cohort_summary, selected_cohort, baseline_mode="median")
    if compare_table.empty:
        st.info("Недостаточно данных для сравнения.")
        return {}, {}
    styled = compare_table.style.format({"selected": "{:,.3f}", "baseline": "{:,.3f}", "delta": "{:+,.3f}"}).apply(
        lambda s: ["color: #2e7d32" if v > 0 else "color: #c62828" if v < 0 else "" for v in s] if s.name == "delta" else [""] * len(s),
        axis=0,
    )
    st.dataframe(styled, use_container_width=True)
    render_item_help("cohort_baseline_compare_help", "Пояснение к сравнению")
    selected_profile = get_selected_cohort_profile(st.session_state["cohort_user_base"], st.session_state["cohort_trips"], selected_cohort)
    baseline_profile = cohort_summary.median(numeric_only=True).to_dict()
    return selected_profile, baseline_profile


def render_selected_cohort_curves(curves: dict) -> None:
    st.subheader("Кривые выбранной когорты")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Retention curve")
        if not curves["retention"].empty:
            st.line_chart(curves["retention"].set_index("month_index")[["selected", "baseline"]], height=260)
        st.caption("Cumulative LTV curve")
        if not curves["ltv"].empty:
            st.line_chart(curves["ltv"].set_index("month_index")[["selected", "baseline"]], height=260)
    with c2:
        st.caption("Cumulative margin curve")
        if not curves["margin"].empty:
            st.line_chart(curves["margin"].set_index("month_index")[["selected", "baseline"]], height=260)
        st.caption("Rides per user curve")
        if not curves["rides"].empty:
            st.line_chart(curves["rides"].set_index("month_index")[["selected", "baseline"]], height=260)


def render_cohort_diagnostics(selected_profile: dict, baseline_profile: dict) -> None:
    st.subheader("Диагностический блок")
    diagnostics = generate_cohort_diagnostics(selected_profile, baseline_profile)
    for line in diagnostics:
        st.markdown(f"- {line}")


def render_cohort_methodology() -> None:
    st.divider()
    render_methodology_footer()


def render(user_mart: pd.DataFrame, trips: pd.DataFrame) -> None:
    render_cohort_header()
    cohort_user_base = build_cohort_user_base(user_mart, trips, pd.DataFrame())
    st.session_state["cohort_user_base"] = cohort_user_base
    st.session_state["cohort_trips"] = trips
    cohort_summary = get_cohort_summary(cohort_user_base)

    filtered_summary, _, mode = render_cohort_filters_info(cohort_summary)
    if filtered_summary.empty:
        st.warning("После локальных фильтров осталось слишком мало данных. Ослабьте ограничения.")
        render_cohort_methodology()
        return

    if filtered_summary["cohort_size"].sum() < 30:
        st.warning("В срезе менее 30 пользователей: возможна высокая волатильность выводов.")
    if len(filtered_summary) < 2:
        st.warning("Для эталонного сравнения желательно минимум 2 когорты.")

    allowed = set(filtered_summary["cohort_month"])
    scoped_base = cohort_user_base.loc[cohort_user_base["cohort_month"].isin(allowed)].copy()
    render_cohort_kpis(filtered_summary)

    matrices = {
        "retention": build_retention_matrix(scoped_base, trips),
        "ltv": build_cumulative_ltv_matrix(scoped_base, trips),
        "margin": build_cumulative_margin_matrix(scoped_base, trips),
        "cancellation": build_cancellation_matrix(scoped_base, trips),
        "promo_share": build_promo_share_matrix(scoped_base, trips),
        "rides": build_rides_per_user_matrix(scoped_base, trips),
        "size": build_cohort_size_matrix(filtered_summary),
    }
    render_cohort_main_matrix(mode, matrices)
    maturity_table = get_cohort_maturity_table(scoped_base, trips["request_ts"].max() if len(trips) else pd.Timestamp.today())
    render_cohort_maturity_table(maturity_table)

    st.subheader("Детализация выбранной когорты")
    selected_cohort = st.selectbox("Выберите когорту", options=filtered_summary["cohort_month"].tolist())
    selected_profile, baseline_profile = render_selected_cohort_compare(filtered_summary, selected_cohort)
    curves = get_selected_cohort_curves(scoped_base, trips, selected_cohort)
    render_selected_cohort_curves(curves)
    render_cohort_diagnostics(selected_profile, baseline_profile)
    render_cohort_methodology()
