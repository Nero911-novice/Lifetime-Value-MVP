
from __future__ import annotations

import streamlit as st

from ..metrics import build_data_model_summary
from ..ui import render_screen_help, render_item_help, render_methodology_footer, info_caption


def render(data):
    st.header("Модель данных")
    render_screen_help("data_model")

    summary = build_data_model_summary(data)
    st.subheader("Состав демонстрационного набора данных")
    st.dataframe(
        summary.rename(
            columns={
                "table_name": "Таблица",
                "rows": "Строки",
                "columns": "Поля",
                "role_in_demo": "Роль в MVP",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Словарь данных")
    info_caption("Словарь показывает, какие поля нужны не только для интерфейса, но и для будущей аналитической модели данных.")
    st.dataframe(data["data_dictionary"], use_container_width=True)
    render_item_help("data_dictionary", "Пояснение к словарю данных")

    st.subheader("Примеры записей")
    tab1, tab2, tab3 = st.tabs(["users", "trips", "marketing_touches"])
    with tab1:
        st.dataframe(data["users"].head(15), use_container_width=True)
    with tab2:
        st.dataframe(data["trips"].head(15), use_container_width=True)
    with tab3:
        st.dataframe(data["marketing_touches"].head(15), use_container_width=True)

    st.subheader("Пограничные случаи в демонстрационных данных")
    edge_cols = st.columns(4)
    edge_cols[0].metric("Не активированы", int((~data["user_mart"]["activated_flag"]).sum()))
    edge_cols[1].metric("Отрицательная маржа поездки", int((data["trips"]["contribution_margin"] < 0).sum()))
    edge_cols[2].metric("Поездки с возвратом", int((data["trips"]["refund_amount"] > 0).sum()))
    edge_cols[3].metric("Маркетинговые касания", int(len(data["marketing_touches"])))

    render_methodology_footer()
