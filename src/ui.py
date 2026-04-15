
from __future__ import annotations

from typing import Iterable
import streamlit as st

from .annotations import ITEM_HELP, SCREEN_HELP
from .methodology import GLOBAL_METHODOLOGY


def render_screen_help(screen_key: str) -> None:
    with st.expander("Справка по экрану", expanded=False):
        st.markdown(SCREEN_HELP[screen_key])


def render_global_filter_help() -> None:
    with st.expander("Пояснение к глобальным фильтрам", expanded=False):
        st.markdown(SCREEN_HELP["global_filters"])


def render_item_help(item_key: str, label: str = "Пояснение") -> None:
    with st.expander(label, expanded=False):
        st.markdown(ITEM_HELP[item_key])


def render_methodology_footer() -> None:
    st.divider()
    with st.expander("Единая методологическая справка", expanded=False):
        st.markdown(GLOBAL_METHODOLOGY)


def format_number(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return f"{value:,}".replace(",", " ")


def format_currency(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{format_number(float(value), digits)} ₽"


def info_caption(text: str) -> None:
    st.caption(text)
