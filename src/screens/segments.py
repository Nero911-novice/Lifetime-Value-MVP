
from __future__ import annotations

import streamlit as st

from ..metrics import (
    build_segment_table,
    build_risk_distribution,
    build_value_distribution,
    build_risk_value_pivot,
)
from ..ui import render_screen_help, render_item_help, render_methodology_footer, info_caption


def render(user_mart):
    st.header("Сегменты")
    render_screen_help("segments")

    risk_dist = build_risk_distribution(user_mart)
    value_dist = build_value_distribution(user_mart)
    segment_table = build_segment_table(user_mart)
    user_pivot = build_risk_value_pivot(user_mart, metric="users")
    cancel_pivot = build_risk_value_pivot(user_mart, metric="avg_cancel_rate")

    top_row_left, top_row_right = st.columns(2)
    with top_row_left:
        st.subheader("Распределение по риску")
        st.bar_chart(risk_dist.set_index("risk_segment")[["users"]], height=260)
        render_item_help("risk_segment", "Пояснение к сегменту риска")

    with top_row_right:
        st.subheader("Распределение по ценности")
        st.bar_chart(value_dist.set_index("value_segment")[["users"]], height=260)
        render_item_help("value_segment", "Пояснение к сегменту ценности")

    st.subheader("Карта риска и ценности: размер сегмента")
    info_caption(
        "Это краткая форма управленческой карты: где находится база пользователей по двум главным осям — риску ухода и фактической ценности."
    )
    st.dataframe(
        user_pivot.style.background_gradient(cmap="Purples").format("{:,.0f}"),
        use_container_width=True,
    )
    render_item_help("risk_value_map", "Пояснение к карте риска и ценности")

    st.subheader("Карта риска и ценности: средняя доля отмен")
    info_caption(
        "Вторая матрица показывает уже не объём сегмента, а его операционное качество. Это полезно для чтения скрытых проблем даже в ценных сегментах."
    )
    st.dataframe(
        cancel_pivot.style.background_gradient(cmap="Oranges").format("{:.1%}"),
        use_container_width=True,
    )

    st.subheader("Сводная таблица сегментов и действий")
    info_caption(
        "Сегменты строятся из трёх осей: риск, ценность и промо-зависимость. Рекомендация — это MVP-логика будущего action layer, а не production-правило."
    )
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
                "avg_cancel_rate": "Средняя доля отмен",
                "recommended_action": "Рекомендованное действие",
            }
        ),
        use_container_width=True,
    )

    render_item_help("promo_band", "Пояснение к промо-зависимости")
    render_item_help("response_rate", "Пояснение к response 7d")

    render_methodology_footer()
