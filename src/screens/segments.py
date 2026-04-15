
from __future__ import annotations

import streamlit as st

from ..metrics import build_segment_table, build_risk_distribution, build_value_distribution
from ..ui import render_screen_help, render_item_help, render_methodology_footer, info_caption


def render(user_mart):
    st.header("Сегменты")
    render_screen_help("segments")

    risk_dist = build_risk_distribution(user_mart)
    value_dist = build_value_distribution(user_mart)
    segment_table = build_segment_table(user_mart)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Распределение по риску")
        st.bar_chart(risk_dist.set_index("risk_segment")[["users"]], height=260)
        render_item_help("risk_segment", "Пояснение к сегменту риска")

    with col2:
        st.subheader("Распределение по ценности")
        st.bar_chart(value_dist.set_index("value_segment")[["users"]], height=260)

    st.subheader("Сводная таблица сегментов")
    info_caption("Сегменты строятся из трех осей: риск, ценность и промо-зависимость. Это MVP-логика для демонстрации будущего decision layer.")
    st.dataframe(
        segment_table.rename(
            columns={
                "risk_segment": "Риск",
                "value_segment": "Ценность",
                "promo_band": "Промо-зависимость",
                "users": "Пользователи",
                "avg_ltv_180": "Средний LTV 180д",
                "avg_trips_90d": "Средние поездки 90д",
                "avg_response_7d": "Средний response 7d",
                "active_90d_share": "Активные 90д",
            }
        ),
        use_container_width=True,
    )

    render_item_help("promo_band", "Пояснение к промо-зависимости")
    render_item_help("response_rate", "Пояснение к response 7d")

    render_methodology_footer()
