from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..metrics import (
    build_segment_user_base,
    get_segment_summary,
    get_segment_map_table,
    get_selected_segment_profile,
    compare_segment_to_baseline,
    get_selected_segment_charts_data,
    generate_segment_diagnostics,
    get_segment_kpis,
)
from ..ui import (
    render_screen_help,
    render_item_help,
    render_methodology_footer,
    format_currency,
    format_number,
    format_percent,
    info_caption,
)
from ..methodology import SEGMENTS_METHODOLOGY


RISK_ORDER = ["Stable / Active", "Cooling", "At risk", "Dormant"]
VALUE_ORDER = ["Low value", "Medium value", "High value"]


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "risk_segment" in df.columns:
        df["risk_segment"] = pd.Categorical(df["risk_segment"], categories=RISK_ORDER, ordered=True)
    if "value_segment" in df.columns:
        df["value_segment"] = pd.Categorical(df["value_segment"], categories=VALUE_ORDER, ordered=True)
    return df.sort_values([c for c in ["value_segment", "risk_segment"] if c in df.columns])


def render_segment_header() -> None:
    st.header("Сегменты")
    st.caption("Диагностика структуры пользовательской базы по осям ценности, риска и промо-зависимости.")
    render_screen_help("segments")


def render_segment_filters_info(segment_user_base: pd.DataFrame) -> pd.DataFrame:
    if segment_user_base.empty:
        return segment_user_base

    c1, c2, c3, c4 = st.columns(4)
    cities = ["Все"] + sorted(segment_user_base["city"].dropna().astype(str).unique().tolist())
    channels = ["Все"] + sorted(segment_user_base["acquisition_channel"].dropna().astype(str).unique().tolist())
    activations = ["Все"] + sorted(segment_user_base["activation_type"].dropna().astype(str).unique().tolist())
    tariffs = ["Все"] + sorted(segment_user_base["preferred_tariff"].dropna().astype(str).unique().tolist())

    city = c1.selectbox("Город (локальный)", cities)
    channel = c2.selectbox("Канал (локальный)", channels)
    activation = c3.selectbox("Тип активации (локальный)", activations)
    tariff = c4.selectbox("Тариф (локальный)", tariffs)

    d1, d2, d3 = st.columns(3)
    promo_options = ["Все"] + sorted(segment_user_base["promo_dependency_segment"].dropna().astype(str).unique().tolist())
    risk_options = ["Все"] + sorted(segment_user_base["risk_segment"].dropna().astype(str).unique().tolist())
    value_options = ["Все"] + sorted(segment_user_base["value_segment"].dropna().astype(str).unique().tolist())

    promo = d1.selectbox("Promo dependency", promo_options)
    risk = d2.selectbox("Risk segment", risk_options)
    value = d3.selectbox("Value segment", value_options)

    tenure_max = int(np.nanmax(segment_user_base["tenure_days"].values)) if segment_user_base["tenure_days"].notna().any() else 0
    tenure_range = st.slider("Tenure, дней", min_value=0, max_value=max(tenure_max, 1), value=(0, max(tenure_max, 1)))

    filtered = segment_user_base.copy()
    if city != "Все":
        filtered = filtered.loc[filtered["city"] == city]
    if channel != "Все":
        filtered = filtered.loc[filtered["acquisition_channel"] == channel]
    if activation != "Все":
        filtered = filtered.loc[filtered["activation_type"] == activation]
    if tariff != "Все":
        filtered = filtered.loc[filtered["preferred_tariff"] == tariff]
    if promo != "Все":
        filtered = filtered.loc[filtered["promo_dependency_segment"] == promo]
    if risk != "Все":
        filtered = filtered.loc[filtered["risk_segment"] == risk]
    if value != "Все":
        filtered = filtered.loc[filtered["value_segment"] == value]

    filtered = filtered.loc[
        filtered["tenure_days"].fillna(tenure_range[0]).between(tenure_range[0], tenure_range[1], inclusive="both")
    ].copy()

    with st.expander("Пояснения к фильтрам и осям сегментации", expanded=False):
        st.markdown("- value_segment, risk_segment и promo_dependency_segment рассчитываются rule-based и служат диагностикой, а не моделью ML.")
        st.markdown("- compound_segment = комбинация risk_segment и value_segment. Промо-зависимость используется как дополнительный слой.")
        st.markdown("- Сравнение с эталоном выполняется относительно медианы сегментов в текущем фильтре.")

    return filtered


def render_segment_kpis(segment_user_base: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    kpis = get_segment_kpis(segment_user_base, segment_summary)
    r1 = st.columns(5)
    r1[0].metric("Пользователи", format_number(kpis["users_count"], 0))
    r1[1].metric("Compound segments", format_number(kpis["compound_segments_count"], 0))
    r1[2].metric("Avg LTV 180d", format_currency(kpis["avg_ltv_180d"], 0))
    r1[3].metric("Total LTV 180d", format_currency(kpis["total_ltv_180d"], 0))
    r1[4].metric("Avg cancellation", format_percent(kpis["avg_cancellation_rate"], 1))
    r2 = st.columns(4)
    r2[0].metric("Avg promo share", format_percent(kpis["avg_promo_trip_share"], 1))
    r2[1].metric("Avg recency", f"{kpis['avg_recency_days']:.1f} д." if pd.notna(kpis["avg_recency_days"]) else "—")
    r2[2].metric("At risk + Dormant", format_percent(kpis["risk_users_share"], 1))
    r2[3].metric("High value share", format_percent(kpis["high_value_users_share"], 1))


def render_segment_main_map(segment_map_table: pd.DataFrame) -> None:
    st.subheader("Карта сегментов риска и ценности")
    info_caption("Ось X: риск, ось Y: ценность. Размер пузыря = users_count. Цвет = выбранный режим метрики.")
    if segment_map_table.empty:
        st.info("Недостаточно данных для построения карты сегментов.")
        return

    color_mode = st.selectbox(
        "Цветовая метрика основной карты",
        ["avg_ltv_180d", "avg_cancellation_rate", "avg_promo_trip_share", "avg_response_rate"],
        format_func=lambda x: {
            "avg_ltv_180d": "Avg LTV 180d",
            "avg_cancellation_rate": "Cancellation rate",
            "avg_promo_trip_share": "Promo trip share",
            "avg_response_rate": "Response rate",
        }[x],
    )

    chart_df = _ordered(segment_map_table.copy())
    st.scatter_chart(
        chart_df,
        x="risk_segment",
        y="value_segment",
        size="users_count",
        color=color_mode,
        use_container_width=True,
        height=340,
    )
    render_item_help("segment_map_help", "Пояснение к карте сегментов")


def render_segment_secondary_map_or_view(segment_map_table: pd.DataFrame) -> None:
    st.subheader("Дополнительный срез карты")
    if segment_map_table.empty:
        return

    mode = st.selectbox(
        "Режим дополнительного среза",
        ["Recommended action", "Avg LTV 180d", "Cancellation rate", "Promo trip share", "Response rate"],
    )
    df = _ordered(segment_map_table.copy())

    if mode == "Recommended action":
        table = df.pivot_table(index="value_segment", columns="risk_segment", values="recommended_action", aggfunc="first")
        st.dataframe(table, use_container_width=True)
    else:
        metric_col = {
            "Avg LTV 180d": "avg_ltv_180d",
            "Cancellation rate": "avg_cancellation_rate",
            "Promo trip share": "avg_promo_trip_share",
            "Response rate": "avg_response_rate",
        }[mode]
        table = df.pivot_table(index="value_segment", columns="risk_segment", values=metric_col)
        styled = table.style.background_gradient(cmap="Blues")
        if metric_col in {"avg_cancellation_rate", "avg_promo_trip_share", "avg_response_rate"}:
            styled = styled.format("{:.1%}")
        else:
            styled = styled.format("{:,.0f}")
        st.dataframe(styled, use_container_width=True)


def render_segment_summary_table(segment_summary: pd.DataFrame) -> str | None:
    st.subheader("Сводная таблица сегментов")
    if segment_summary.empty:
        st.info("Нет сегментов в текущем срезе.")
        return None

    view = _ordered(segment_summary.copy())
    view = view[
        [
            "compound_segment", "users_count", "users_share", "avg_ltv_180d", "total_ltv_180d",
            "avg_margin_per_completed_order", "avg_cancellation_rate", "avg_promo_trip_share",
            "avg_recency_days", "avg_rides_last_90d", "avg_response_rate", "recommended_action",
        ]
    ]
    st.dataframe(
        view.rename(columns={
            "compound_segment": "Segment",
            "users_count": "Users",
            "users_share": "Users share",
            "avg_ltv_180d": "Avg LTV 180d",
            "total_ltv_180d": "Total LTV 180d",
            "avg_margin_per_completed_order": "Avg margin / completed order",
            "avg_cancellation_rate": "Cancellation rate",
            "avg_promo_trip_share": "Promo trip share",
            "avg_recency_days": "Avg recency",
            "avg_rides_last_90d": "Avg rides 90d",
            "avg_response_rate": "Avg response rate",
            "recommended_action": "Recommended action",
        }),
        use_container_width=True,
    )

    return st.selectbox("Выберите compound segment", options=view["compound_segment"].tolist())


def render_selected_segment_profile(selected_profile: dict) -> None:
    st.subheader("Карточка выбранного сегмента")
    if not selected_profile:
        st.info("Выберите сегмент для детализации.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", format_number(selected_profile.get("users_count"), 0))
    c2.metric("Users share", format_percent(selected_profile.get("users_share"), 1))
    c3.metric("Avg LTV 180d", format_currency(selected_profile.get("avg_ltv_180d"), 0))
    c4.metric("Total LTV 180d", format_currency(selected_profile.get("total_ltv_180d"), 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg margin/order", format_currency(selected_profile.get("avg_margin_per_completed_order"), 1))
    c6.metric("Cancellation", format_percent(selected_profile.get("cancellation_rate"), 1))
    c7.metric("Promo share", format_percent(selected_profile.get("promo_trip_share"), 1))
    c8.metric("Response rate", format_percent(selected_profile.get("avg_response_rate"), 1))

    st.caption(
        f"Risk: {selected_profile.get('risk_segment')} · Value: {selected_profile.get('value_segment')} · "
        f"Dominant promo dependency: {selected_profile.get('dominant_promo_segment')} · "
        f"Recommended action (rule-based): {selected_profile.get('recommended_action')}"
    )


def render_selected_segment_compare(segment_summary: pd.DataFrame, selected_segment: str) -> tuple[dict, dict]:
    st.subheader("Сравнение сегмента с эталоном (медиана)")
    compare_df = compare_segment_to_baseline(segment_summary, selected_segment, baseline_mode="median")
    if compare_df.empty:
        st.info("Недостаточно данных для сравнения с эталоном.")
        return {}, {}

    styled = compare_df.style.format({"selected": "{:,.3f}", "baseline": "{:,.3f}", "delta": "{:+,.3f}"}).apply(
        lambda s: ["color: #2e7d32" if v > 0 else "color: #c62828" if v < 0 else "" for v in s] if s.name == "delta" else [""] * len(s),
        axis=0,
    )
    st.dataframe(styled, use_container_width=True)
    render_item_help("segment_baseline_compare_help", "Пояснение к сравнению с эталоном")

    selected_row = segment_summary.loc[segment_summary["compound_segment"] == selected_segment].iloc[0].to_dict()
    baseline_row = segment_summary.median(numeric_only=True).to_dict()
    baseline_profile = {
        "avg_ltv_180d": baseline_row.get("avg_ltv_180d", np.nan),
        "total_ltv_180d": baseline_row.get("total_ltv_180d", np.nan),
        "cancellation_rate": baseline_row.get("avg_cancellation_rate", np.nan),
        "promo_trip_share": baseline_row.get("avg_promo_trip_share", np.nan),
        "avg_recency_days": baseline_row.get("avg_recency_days", np.nan),
        "avg_rides_last_90d": baseline_row.get("avg_rides_last_90d", np.nan),
    }
    selected_profile = {
        "avg_ltv_180d": selected_row.get("avg_ltv_180d", np.nan),
        "total_ltv_180d": selected_row.get("total_ltv_180d", np.nan),
        "cancellation_rate": selected_row.get("avg_cancellation_rate", np.nan),
        "promo_trip_share": selected_row.get("avg_promo_trip_share", np.nan),
        "avg_recency_days": selected_row.get("avg_recency_days", np.nan),
        "avg_rides_last_90d": selected_row.get("avg_rides_last_90d", np.nan),
    }
    return selected_profile, baseline_profile


def render_selected_segment_charts(charts: dict[str, pd.DataFrame]) -> None:
    st.subheader("Профиль выбранного сегмента")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Ключевые метрики: сегмент vs baseline")
        if not charts["key_metrics"].empty:
            st.bar_chart(charts["key_metrics"].set_index("metric")[["selected", "baseline"]], height=280)
        st.caption("Recency vs rides profile")
        if not charts["recency_rides"].empty:
            st.bar_chart(charts["recency_rides"].set_index("profile")[["recency_days", "rides_last_90d"]], height=220)
    with c2:
        st.caption("Promo vs margin profile")
        if not charts["promo_margin"].empty:
            st.bar_chart(charts["promo_margin"].set_index("profile")[["promo_trip_share", "avg_margin_per_completed_order"]], height=220)
        st.caption("Cancellation vs value profile")
        if not charts["cancel_value"].empty:
            st.bar_chart(charts["cancel_value"].set_index("profile")[["cancellation_rate", "avg_ltv_180d"]], height=220)


def render_segment_diagnostics(selected_profile: dict, baseline_profile: dict) -> None:
    st.subheader("Диагностический блок")
    notes = generate_segment_diagnostics(selected_profile, baseline_profile)
    for note in notes:
        st.markdown(f"- {note}")


def render_segment_methodology() -> None:
    st.subheader("Методологическая рамка экрана")
    with st.expander("Почему экран сегментов устроен именно так", expanded=False):
        st.markdown(SEGMENTS_METHODOLOGY)
    render_methodology_footer()


def _render_context_help() -> None:
    with st.expander("Контекстная справка по метрикам и сегментам", expanded=False):
        for key, label in [
            ("segment_value_help", "Value segment"),
            ("segment_risk_help", "Risk segment"),
            ("segment_promo_dependency_help", "Promo dependency"),
            ("compound_segment_help", "Compound segment"),
            ("segment_avg_ltv_help", "Avg LTV 180d"),
            ("segment_total_ltv_help", "Total LTV 180d"),
            ("segment_margin_help", "Avg margin per completed order"),
            ("segment_cancellation_help", "Cancellation rate"),
            ("segment_promo_share_help", "Promo trip share"),
            ("segment_recency_help", "Recency"),
            ("segment_rides_90d_help", "Rides last 90d"),
            ("segment_recommended_action_help", "Recommended action"),
        ]:
            st.markdown(f"**{label}**")
            st.caption(st.session_state.get(f"help_{key}", ""))
            render_item_help(key, "Подробнее")


def render(user_mart: pd.DataFrame, trips: pd.DataFrame | None = None, touches: pd.DataFrame | None = None) -> None:
    render_segment_header()
    segment_user_base = build_segment_user_base(user_mart, trips, touches)

    if segment_user_base.empty or len(segment_user_base) < 5:
        st.warning("После фильтрации слишком мало пользователей для надёжной сегментной диагностики.")
        render_segment_methodology()
        return

    filtered = render_segment_filters_info(segment_user_base)
    if filtered.empty:
        st.warning("Локальные фильтры исключили всех пользователей. Измените условия отбора.")
        render_segment_methodology()
        return

    if len(filtered) < 30:
        st.warning("В срезе менее 30 пользователей: сегментные выводы могут быть волатильными.")

    segment_summary = get_segment_summary(filtered)
    segment_map_table = get_segment_map_table(filtered)

    render_segment_kpis(filtered, segment_summary)
    render_segment_main_map(segment_map_table)
    render_segment_secondary_map_or_view(segment_map_table)

    selected_segment = render_segment_summary_table(segment_summary)
    if selected_segment:
        selected_profile_full = get_selected_segment_profile(filtered, selected_segment)
        render_selected_segment_profile(selected_profile_full)
        selected_profile, baseline_profile = render_selected_segment_compare(segment_summary, selected_segment)
        charts = get_selected_segment_charts_data(filtered, selected_segment)
        render_selected_segment_charts(charts)
        render_segment_diagnostics(selected_profile, baseline_profile)

    _render_context_help()
    render_segment_methodology()
