
from __future__ import annotations

import streamlit as st

from ..metrics import compute_cohort_matrices, build_selected_cohort_curves
from ..ui import (
    render_screen_help,
    render_item_help,
    render_methodology_footer,
    info_caption,
    format_percent,
    format_currency,
    format_number,
)


def render(user_mart, trips):
    st.header("Когорты")
    render_screen_help("cohorts")

    max_age = st.slider(
        "Максимальный возраст когорты, месяцев",
        min_value=6,
        max_value=18,
        value=12,
        step=1,
        help="Ограничивает глубину когортных матриц по месяцам жизни.",
    )

    matrices = compute_cohort_matrices(user_mart, trips, max_age_months=max_age)
    cohort_summary = matrices["cohort_summary"].copy()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    mature = cohort_summary.loc[cohort_summary["maturity_months"] >= 6]
    kpi1.metric("Когорт активации", format_number(len(cohort_summary), 0))
    kpi2.metric("Активированных пользователей", format_number(cohort_summary["activated_users"].sum(), 0))
    kpi3.metric("Средний LTV 180д по зрелым когортам", format_currency(mature["avg_ltv_180"].mean() if len(mature) else 0, 0))
    kpi4.metric("Средняя активность 90д по зрелым когортам", format_percent(mature["active_90d_share"].mean() if len(mature) else 0, 1))

    st.subheader("Сводка по когортам")
    info_caption(
        "Сводная таблица нужна для управленческого чтения качества когорт: размер, зрелость, LTV, активность и отмены рассматриваются одновременно."
    )
    st.dataframe(
        cohort_summary.rename(
            columns={
                "cohort_month": "Когорта",
                "activated_users": "Активированные пользователи",
                "avg_ltv_180": "Средний LTV 180д",
                "active_90d_share": "Активные 90д",
                "avg_orders": "Средние созданные заказы",
                "avg_completed_orders": "Средние завершенные заказы",
                "avg_cancel_rate": "Средняя доля отмен",
                "avg_cac": "Средний CAC",
                "maturity_months": "Зрелость, месяцев",
            }
        ),
        use_container_width=True,
    )
    render_item_help("cohort_summary", "Пояснение к сводке по когортам")

    st.subheader("Когортный retention")
    info_caption("Строка — когорта первой поездки. Столбцы M0, M1 ... — месяцы жизни когорты.")
    st.dataframe(
        matrices["retention_matrix"].style.format("{:.1%}").background_gradient(cmap="Blues"),
        use_container_width=True,
    )
    render_item_help("retention_matrix", "Пояснение к retention")

    st.subheader("Когортный накопленный LTV")
    info_caption("Матрица показывает среднюю накопленную маржу на активированного пользователя когорты.")
    st.dataframe(
        matrices["cohort_ltv_matrix"].style.format("{:,.0f} ₽").background_gradient(cmap="Greens"),
        use_container_width=True,
    )
    render_item_help("cohort_ltv_matrix", "Пояснение к когортному LTV")

    cohort_options = cohort_summary["cohort_month"].tolist()
    selected_cohort = st.selectbox(
        "Детализация по когорте",
        options=cohort_options,
        help="Позволяет разобрать поведение одной когорты по месяцам жизни и увидеть, как сочетаются retention и накопленный LTV.",
    )
    selected_curve = build_selected_cohort_curves(matrices, selected_cohort)
    selected_row = cohort_summary.loc[cohort_summary["cohort_month"] == selected_cohort].iloc[0]

    det1, det2, det3, det4 = st.columns(4)
    det1.metric("Пользователи в когорте", format_number(selected_row["activated_users"], 0))
    det2.metric("Зрелость", f"{int(selected_row['maturity_months'])} мес.")
    det3.metric("Средний LTV 180д", format_currency(selected_row["avg_ltv_180"], 0))
    det4.metric("Средняя доля отмен", format_percent(selected_row["avg_cancel_rate"], 1))

    curve_left, curve_right = st.columns(2)
    with curve_left:
        st.caption("Retention выбранной когорты")
        if not selected_curve.empty:
            st.line_chart(selected_curve.set_index("age_month")[["retention"]], height=280)
    with curve_right:
        st.caption("Накопленный LTV выбранной когорты")
        if not selected_curve.empty:
            st.line_chart(selected_curve.set_index("age_month")[["cumulative_ltv"]], height=280)

    st.subheader("Зрелость когорт")
    st.dataframe(
        matrices["cohort_maturity"].rename(
            columns={
                "cohort_month": "Когорта",
                "activated_users": "Активированные пользователи",
                "maturity_months": "Зрелость, месяцев",
            }
        ),
        use_container_width=True,
    )

    render_methodology_footer()
