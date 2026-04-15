
from __future__ import annotations

import streamlit as st

from ..metrics import compute_overview_metrics, build_overview_charts
from ..ui import render_screen_help, render_item_help, render_methodology_footer, format_currency, format_number, info_caption


def render(user_mart):
    st.header("Обзор")
    render_screen_help("overview")

    metrics = compute_overview_metrics(user_mart)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Пользователи", format_number(metrics["total_users"], 0))
    col2.metric("Активация", f'{metrics["activation_rate"]:.1%}')
    col3.metric("Исторический LTV 180д", format_currency(metrics["ltv_180_mean"], 0))
    col4.metric("LTV/CAC", "—" if metrics["ltv_cac_ratio"] != metrics["ltv_cac_ratio"] else f'{metrics["ltv_cac_ratio"]:.2f}')

    col5, col6, col7 = st.columns(3)
    col5.metric("Завершенные поездки", format_number(metrics["completed_trips"], 0))
    col6.metric("Средняя маржа поездки", format_currency(metrics["avg_trip_margin"], 0))
    col7.metric("Активные за 90 дней", f'{metrics["active_90d_share"]:.1%}')

    render_item_help("hist_ltv_180", "Пояснение к историческому LTV 180д")
    render_item_help("contribution_margin", "Пояснение к contribution margin")
    render_item_help("active_90d_share", "Пояснение к доле активных за 90 дней")
    render_item_help("ltv_cac_ratio", "Пояснение к LTV/CAC")

    charts = build_overview_charts(user_mart)

    st.subheader("Динамика когорт активации")
    info_caption("Ниже показано, как меняется средний исторический LTV 180д и доля активных за 90 дней по когортам первой поездки.")
    st.line_chart(
        charts["cohort_trend"].set_index("cohort_month")[["avg_ltv_180", "active_90d_share"]],
        height=300,
    )

    st.subheader("Сравнение каналов привлечения")
    info_caption("Канал — один из ключевых управленческих срезов: здесь сопоставляются масштаб, активация и историческая ценность.")
    st.dataframe(
        charts["channel_summary"].rename(
            columns={
                "acquisition_channel": "Канал",
                "users": "Пользователи",
                "activation_rate": "Активация",
                "avg_ltv_180": "Средний LTV 180д",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Городской срез")
    st.bar_chart(
        charts["city_summary"].set_index("home_city")[["avg_ltv_180", "active_90d_share"]],
        height=280,
    )

    render_methodology_footer()
