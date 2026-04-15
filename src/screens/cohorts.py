
from __future__ import annotations

import importlib.util

import streamlit as st

from ..metrics import compute_cohort_matrices
from ..ui import render_screen_help, render_item_help, render_methodology_footer, info_caption


def _style_with_optional_gradient(df, value_format: str, cmap: str):
    styler = df.style.format(value_format)
    if importlib.util.find_spec("matplotlib") is not None:
        styler = styler.background_gradient(cmap=cmap)
    return styler


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

    st.subheader("Когортный retention")
    info_caption("Строка — когорта первой поездки. Столбцы M0, M1 ... — месяцы жизни когорты.")
    st.dataframe(
        _style_with_optional_gradient(matrices["retention_matrix"], "{:.1%}", "Blues"),
        use_container_width=True,
    )
    render_item_help("retention_matrix", "Пояснение к retention")

    st.subheader("Когортный накопленный LTV")
    info_caption("Матрица показывает среднюю накопленную маржу на активированного пользователя когорты.")
    st.dataframe(
        _style_with_optional_gradient(matrices["cohort_ltv_matrix"], "{:,.0f} ₽", "Greens"),
        use_container_width=True,
    )
    render_item_help("cohort_ltv_matrix", "Пояснение к когортному LTV")

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
