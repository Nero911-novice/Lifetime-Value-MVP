
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data_loader import get_data_bundle
from src.metrics import apply_common_filters, filter_related_tables
from src.ui import render_global_filter_help
from src.screens import overview, cohorts, segments, user_profile, data_model


st.set_page_config(
    page_title="LTV Demo MVP · Ride-Hailing",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data(show_spinner=False)
def load_bundle():
    return get_data_bundle(DATA_DIR)


def main() -> None:
    data = load_bundle()
    user_mart = data["user_mart"]

    st.title("LTV Demo MVP для райд-хейлинга")
    st.caption(
        "Демонстрационный аналитический прототип: показывает будущую структуру LTV-продукта, связь между данными и метриками, а также управленческую логику экранов."
    )

    with st.sidebar:
        st.header("Навигация")
        page = st.radio(
            "Раздел",
            options=[
                "Обзор",
                "Когорты",
                "Сегменты",
                "Карточка пользователя",
                "Модель данных",
            ],
        )

        st.header("Глобальные фильтры")
        cities = ["Все"] + sorted(user_mart["home_city"].dropna().unique().tolist())
        channels = ["Все"] + sorted(user_mart["acquisition_channel"].dropna().unique().tolist())
        tariffs = ["Все"] + sorted(user_mart["preferred_tariff"].dropna().unique().tolist())
        activation_types = ["Все"] + sorted(user_mart["activation_type"].dropna().unique().tolist())

        selected_city = st.selectbox("Город", cities, help="Фильтр по домашнему городу пользователя.")
        selected_channel = st.selectbox("Канал привлечения", channels, help="Фильтр по первичному каналу привлечения.")
        selected_tariff = st.selectbox("Предпочитаемый тариф", tariffs, help="Фильтр по базовому тарифному профилю.")
        selected_activation = st.selectbox("Тип активации", activation_types, help="Фильтр по сценарию первой активации.")

        render_global_filter_help()

    filtered_user_mart = apply_common_filters(
        user_mart=user_mart,
        city=selected_city,
        channel=selected_channel,
        tariff=selected_tariff,
        activation_type=selected_activation,
    )
    filtered_related = filter_related_tables(data, filtered_user_mart)
    filtered_related["user_mart"] = filtered_user_mart

    if page == "Обзор":
        overview.render(filtered_user_mart, filtered_related["trips"])
    elif page == "Когорты":
        cohorts.render(filtered_user_mart, filtered_related["trips"])
    elif page == "Сегменты":
        segments.render(filtered_user_mart)
    elif page == "Карточка пользователя":
        user_profile.render(filtered_related)
    elif page == "Модель данных":
        data_model.render(filtered_related)


if __name__ == "__main__":
    main()
