from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..metrics import (
    build_segment_user_base,
    compare_segment_to_baseline,
    generate_segment_diagnostics,
    get_segment_kpis,
    get_segment_map_table,
    get_segment_summary,
    get_selected_segment_charts_data,
    get_selected_segment_profile,
)
from ..methodology import SEGMENTS_METHODOLOGY
from ..ui import (
    format_currency,
    format_number,
    format_percent,
    render_methodology_footer,
    render_screen_help,
)

RISK_ORDER = ["Stable / Active", "Cooling", "At risk", "Dormant"]
VALUE_ORDER = ["Low value", "Medium value", "High value"]

RISK_LABELS = {
    "Stable / Active": "Стабильные / активные",
    "Cooling": "Остывающие",
    "At risk": "В зоне риска",
    "Dormant": "Спящие",
}
VALUE_LABELS = {
    "Low value": "Низкая ценность",
    "Medium value": "Средняя ценность",
    "High value": "Высокая ценность",
}
PROMO_LABELS = {
    "Low promo dependency": "Низкая промо-зависимость",
    "Medium promo dependency": "Средняя промо-зависимость",
    "High promo dependency": "Высокая промо-зависимость",
}
ACTION_LABELS = {
    "Protect / Retain": "Удерживать",
    "Reactivate": "Реактивировать",
    "Stimulate carefully": "Стимулировать осторожно",
    "Limit incentives": "Ограничить субсидии",
    "Observe / No immediate action": "Наблюдать без немедленного действия",
}


def _localize(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    if "risk_segment" in localized.columns:
        localized["risk_segment_ru"] = localized["risk_segment"].map(RISK_LABELS).fillna("недостаточно данных")
    if "value_segment" in localized.columns:
        localized["value_segment_ru"] = localized["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
    if "promo_dependency_segment" in localized.columns:
        localized["promo_dependency_segment_ru"] = localized["promo_dependency_segment"].map(PROMO_LABELS).fillna("недостаточно данных")
    if "recommended_action" in localized.columns:
        localized["recommended_action_ru"] = localized["recommended_action"].map(ACTION_LABELS).fillna("Наблюдать без немедленного действия")
    if "dominant_promo_segment" in localized.columns:
        localized["dominant_promo_segment_ru"] = localized["dominant_promo_segment"].map(PROMO_LABELS).fillna("недостаточно данных")
    if "compound_segment" in localized.columns:
        localized["compound_segment_ru"] = localized["risk_segment"].map(RISK_LABELS).fillna("недостаточно данных") + " × " + localized["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
    return localized


def render_segment_header() -> None:
    st.header("Сегменты")
    st.caption("Структура базы по ценности, риску и промо-зависимости для управленческой диагностики.")
    render_screen_help("segments")


def render_segment_filters_info(segment_user_base: pd.DataFrame) -> pd.DataFrame:
    localized = _localize(segment_user_base)
    c1, c2, c3, c4 = st.columns(4)

    city = c1.selectbox("Город", ["Все"] + sorted(localized["city"].dropna().astype(str).unique().tolist()))
    channel = c2.selectbox("Канал привлечения", ["Все"] + sorted(localized["acquisition_channel"].dropna().astype(str).unique().tolist()))
    activation = c3.selectbox("Тип активации", ["Все"] + sorted(localized["activation_type"].dropna().astype(str).unique().tolist()))
    tariff = c4.selectbox("Тариф", ["Все"] + sorted(localized["preferred_tariff"].dropna().astype(str).unique().tolist()))

    d1, d2, d3 = st.columns(3)
    promo = d1.selectbox("Промо-зависимость", ["Все"] + list(PROMO_LABELS.values()))
    risk = d2.selectbox("Сегмент риска", ["Все"] + [RISK_LABELS[x] for x in RISK_ORDER])
    value = d3.selectbox("Сегмент ценности", ["Все"] + [VALUE_LABELS[x] for x in VALUE_ORDER])

    tenure_max = int(np.nanmax(localized["tenure_days"].values)) if localized["tenure_days"].notna().any() else 365
    tenure_range = st.slider("Срок жизни, дней", min_value=0, max_value=max(tenure_max, 1), value=(0, max(tenure_max, 1)))

    filtered = localized.copy()
    if city != "Все":
        filtered = filtered.loc[filtered["city"] == city]
    if channel != "Все":
        filtered = filtered.loc[filtered["acquisition_channel"] == channel]
    if activation != "Все":
        filtered = filtered.loc[filtered["activation_type"] == activation]
    if tariff != "Все":
        filtered = filtered.loc[filtered["preferred_tariff"] == tariff]
    if promo != "Все":
        filtered = filtered.loc[filtered["promo_dependency_segment_ru"] == promo]
    if risk != "Все":
        filtered = filtered.loc[filtered["risk_segment_ru"] == risk]
    if value != "Все":
        filtered = filtered.loc[filtered["value_segment_ru"] == value]
    filtered = filtered.loc[filtered["tenure_days"].fillna(tenure_range[0]).between(tenure_range[0], tenure_range[1], inclusive="both")]

    with st.expander("Как читать сегментацию", expanded=False):
        st.markdown("- Риск: давность и интенсивность поездок в 30/90 дней.")
        st.markdown("- Ценность: исторический LTV 180 дней с поправкой на активность.")
        st.markdown("- Промо-зависимость: сочетание доли промо-поездок и response rate.")
        st.markdown("- Рекомендованное действие — rule-based аналитическая подсказка, а не decision engine.")

    return filtered


def render_segment_kpis(segment_user_base: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    kpis = get_segment_kpis(segment_user_base, segment_summary)
    r1 = st.columns(5)
    r1[0].metric("Пользователи в срезе", format_number(kpis["users_count"], 0))
    r1[1].metric("Непустые комбинированные сегменты", format_number(kpis["compound_segments_count"], 0))
    r1[2].metric("Средний LTV 180 дней", format_currency(kpis["avg_ltv_180d"], 0))
    r1[3].metric("Суммарный LTV 180 дней", format_currency(kpis["total_ltv_180d"], 0))
    r1[4].metric("Средняя доля отмен", format_percent(kpis["avg_cancellation_rate"], 1))

    r2 = st.columns(4)
    r2[0].metric("Средняя доля промо-поездок", format_percent(kpis["avg_promo_trip_share"], 1))
    r2[1].metric("Средний recency", f"{format_number(kpis['avg_recency_days'], 1)} дн." if pd.notna(kpis["avg_recency_days"]) else "недостаточно данных")
    r2[2].metric("Доля пользователей в зоне риска", format_percent(kpis["risk_users_share"], 1))
    r2[3].metric("Доля пользователей высокой ценности", format_percent(kpis["high_value_users_share"], 1))


def render_segment_heatmap(segment_map_table: pd.DataFrame) -> None:
    st.subheader("Карта сегментов риска и ценности")
    if segment_map_table.empty:
        st.info("Недостаточно данных для построения карты сегментов.")
        return

    localized = _localize(segment_map_table)
    metric_mode = st.selectbox(
        "Режим карты",
        [
            "users_count",
            "users_share",
            "avg_ltv_180d",
            "total_ltv_180d",
            "avg_cancellation_rate",
            "avg_promo_trip_share",
            "avg_response_rate",
        ],
        format_func=lambda x: {
            "users_count": "Размер сегмента",
            "users_share": "Доля базы",
            "avg_ltv_180d": "Средний LTV 180 дней",
            "total_ltv_180d": "Суммарный LTV 180 дней",
            "avg_cancellation_rate": "Доля отмен",
            "avg_promo_trip_share": "Доля промо-поездок",
            "avg_response_rate": "Средний response rate",
        }[x],
    )

    pivot = localized.pivot_table(
        index="risk_segment_ru",
        columns="value_segment_ru",
        values=metric_mode,
        aggfunc="mean",
    ).reindex(index=[RISK_LABELS[x] for x in RISK_ORDER], columns=[VALUE_LABELS[x] for x in VALUE_ORDER])

    formatted = pivot.copy()
    if metric_mode in {"users_share", "avg_cancellation_rate", "avg_promo_trip_share", "avg_response_rate"}:
        formatted = formatted.map(lambda x: format_percent(x, 1) if pd.notna(x) else "не рассчитывается в текущем срезе")
    elif metric_mode in {"avg_ltv_180d", "total_ltv_180d"}:
        formatted = formatted.map(lambda x: format_currency(x, 0) if pd.notna(x) else "не рассчитывается в текущем срезе")
    else:
        formatted = formatted.map(lambda x: format_number(x, 0) if pd.notna(x) else "0")

    st.dataframe(formatted, use_container_width=True)


def render_segment_summary_table(segment_summary: pd.DataFrame) -> str | None:
    st.subheader("Сводная таблица сегментов")
    if segment_summary.empty:
        st.info("Нет сегментов в текущем срезе.")
        return None

    localized = _localize(segment_summary.copy()).sort_values(["users_count", "total_ltv_180d"], ascending=[False, False])
    table = localized[
        [
            "compound_segment",
            "compound_segment_ru",
            "users_count",
            "users_share",
            "avg_ltv_180d",
            "total_ltv_180d",
            "avg_margin_per_completed_order",
            "avg_cancellation_rate",
            "avg_promo_trip_share",
            "avg_recency_days",
            "avg_rides_last_90d",
            "avg_response_rate",
            "recommended_action_ru",
        ]
    ].rename(
        columns={
            "compound_segment_ru": "Комбинированный сегмент",
            "users_count": "Пользователи",
            "users_share": "Доля пользователей",
            "avg_ltv_180d": "Средний LTV 180 дней",
            "total_ltv_180d": "Суммарный LTV 180 дней",
            "avg_margin_per_completed_order": "Средняя маржа завершённой поездки",
            "avg_cancellation_rate": "Доля отмен",
            "avg_promo_trip_share": "Доля промо-поездок",
            "avg_recency_days": "Средний recency",
            "avg_rides_last_90d": "Среднее число поездок за 90 дней",
            "avg_response_rate": "Средний response rate",
            "recommended_action_ru": "Рекомендованное действие",
        }
    )

    display = table.drop(columns=["compound_segment"]).copy()
    for col in ["Доля пользователей", "Доля отмен", "Доля промо-поездок", "Средний response rate"]:
        display[col] = display[col].map(lambda x: format_percent(x, 1) if pd.notna(x) else "не рассчитывается в текущем срезе")
    for col in ["Средний LTV 180 дней", "Суммарный LTV 180 дней", "Средняя маржа завершённой поездки"]:
        display[col] = display[col].map(lambda x: format_currency(x, 0) if pd.notna(x) else "нет завершённых поездок")
    display["Средний recency"] = display["Средний recency"].map(lambda x: f"{format_number(x, 1)} дн." if pd.notna(x) else "нет завершённых поездок")
    display["Среднее число поездок за 90 дней"] = display["Среднее число поездок за 90 дней"].map(lambda x: format_number(x, 2) if pd.notna(x) else "недостаточно данных")

    st.dataframe(display, use_container_width=True)

    options = table[["compound_segment", "Комбинированный сегмент"]]
    selected_label = st.selectbox("Выберите комбинированный сегмент", options["Комбинированный сегмент"].tolist())
    return options.loc[options["Комбинированный сегмент"] == selected_label, "compound_segment"].iloc[0]


def render_selected_segment_profile(selected_profile: dict) -> None:
    st.subheader("Карточка выбранного сегмента")
    if not selected_profile:
        st.info("Выберите сегмент для детализации.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Размер сегмента", format_number(selected_profile.get("users_count"), 0))
    c2.metric("Доля базы", format_percent(selected_profile.get("users_share"), 1))
    c3.metric("Средний LTV 180 дней", format_currency(selected_profile.get("avg_ltv_180d"), 0))
    c4.metric("Суммарный LTV 180 дней", format_currency(selected_profile.get("total_ltv_180d"), 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Средняя маржа завершённой поездки", format_currency(selected_profile.get("avg_margin_per_completed_order"), 1))
    c6.metric("Доля отмен", format_percent(selected_profile.get("cancellation_rate"), 1))
    c7.metric("Доля промо-поездок", format_percent(selected_profile.get("promo_trip_share"), 1))
    c8.metric("Средний response rate", format_percent(selected_profile.get("avg_response_rate"), 1))

    c9, c10 = st.columns(2)
    c9.metric("Средний recency", f"{format_number(selected_profile.get('avg_recency_days'), 1)} дн." if pd.notna(selected_profile.get("avg_recency_days")) else "нет завершённых поездок")
    c10.metric("Среднее число поездок за 90 дней", format_number(selected_profile.get("avg_rides_last_90d"), 2))

    risk = RISK_LABELS.get(selected_profile.get("risk_segment"), "недостаточно данных")
    value = VALUE_LABELS.get(selected_profile.get("value_segment"), "недостаточно данных")
    promo = PROMO_LABELS.get(selected_profile.get("dominant_promo_segment"), "недостаточно данных")
    action = ACTION_LABELS.get(selected_profile.get("recommended_action"), "Наблюдать без немедленного действия")
    st.caption(f"Риск: {risk} · Ценность: {value} · Доминирующая промо-зависимость: {promo} · Рекомендованное действие: {action}")


def render_selected_segment_compare(segment_summary: pd.DataFrame, selected_segment: str) -> tuple[dict, dict]:
    st.subheader("Сравнение с эталоном")
    if segment_summary["compound_segment"].nunique() < 2:
        st.warning("Для сравнения с эталоном нужно минимум два непустых сегмента в текущем срезе.")
        return {}, {}

    compare_df = compare_segment_to_baseline(segment_summary, selected_segment, baseline_mode="median")
    if compare_df.empty:
        st.warning("Сравнение не рассчитывается в текущем срезе.")
        return {}, {}

    compare_df["Метрика"] = compare_df["metric"]
    compare_df["Выбранный сегмент"] = compare_df["selected"].map(lambda x: format_number(x, 3) if pd.notna(x) else "не рассчитывается")
    compare_df["Эталон (медиана)"] = compare_df["baseline"].map(lambda x: format_number(x, 3) if pd.notna(x) else "не рассчитывается")
    compare_df["Отклонение"] = compare_df["delta"].map(lambda x: f"{format_number(x, 3)}" if pd.notna(x) else "не рассчитывается")
    st.dataframe(compare_df[["Метрика", "Выбранный сегмент", "Эталон (медиана)", "Отклонение"]], use_container_width=True)

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
    st.subheader("Профиль сегмента")
    if charts["key_metrics"].empty:
        st.info("Недостаточно разнообразия сегментов для профильных графиков.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Ключевые метрики: сегмент vs эталон")
        st.bar_chart(charts["key_metrics"].set_index("metric")[["selected", "baseline"]], height=260)
        st.caption("Профиль риск vs активность")
        if not charts["recency_rides"].empty:
            st.bar_chart(charts["recency_rides"].set_index("profile")[["recency_days", "rides_last_90d"]], height=220)

    with c2:
        st.caption("Профиль ценность vs промо")
        if not charts["promo_margin"].empty:
            st.bar_chart(charts["promo_margin"].set_index("profile")[["promo_trip_share", "avg_margin_per_completed_order"]], height=220)
        st.caption("Профиль отмены vs ценность")
        if not charts["cancel_value"].empty:
            st.bar_chart(charts["cancel_value"].set_index("profile")[["cancellation_rate", "avg_ltv_180d"]], height=220)


def render_segment_diagnostics(selected_profile: dict, baseline_profile: dict) -> None:
    st.subheader("Диагностический блок")
    notes = generate_segment_diagnostics(selected_profile, baseline_profile)
    for note in notes:
        st.markdown(f"- {note}")


def render_segment_methodology() -> None:
    st.subheader("Методологическая справка")
    with st.expander("Как устроен экран сегментов", expanded=False):
        st.markdown(SEGMENTS_METHODOLOGY)
    render_methodology_footer()


def render(user_mart: pd.DataFrame, trips: pd.DataFrame | None = None, touches: pd.DataFrame | None = None) -> None:
    render_segment_header()
    segment_user_base = build_segment_user_base(user_mart, trips, touches)

    if segment_user_base.empty:
        st.warning("Недостаточно данных для сегментации в текущем срезе.")
        render_segment_methodology()
        return

    filtered = render_segment_filters_info(segment_user_base)
    if filtered.empty:
        st.warning("После применения фильтров пользователей не осталось. Измените условия отбора.")
        render_segment_methodology()
        return

    if len(filtered) < 30:
        st.warning("В срезе менее 30 пользователей: выводы по сегментам могут быть волатильными.")

    segment_summary = get_segment_summary(filtered)
    segment_map_table = get_segment_map_table(filtered)

    render_segment_kpis(filtered, segment_summary)
    render_segment_heatmap(segment_map_table)

    selected_segment = render_segment_summary_table(segment_summary)
    if selected_segment:
        selected_profile_full = get_selected_segment_profile(filtered, selected_segment)
        render_selected_segment_profile(selected_profile_full)
        selected_profile, baseline_profile = render_selected_segment_compare(segment_summary, selected_segment)
        charts = get_selected_segment_charts_data(filtered, selected_segment)
        render_selected_segment_charts(charts)
        if selected_profile and baseline_profile:
            render_segment_diagnostics(selected_profile, baseline_profile)

    render_segment_methodology()
