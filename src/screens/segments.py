from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from ..metrics import (
    build_segment_user_base,
    compare_segment_to_baseline,
    generate_segment_diagnostics,
    get_ltv_concentration_by_value_segment,
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
from ..segment_labels import (
    ACTION_LABELS,
    PROMO_LABELS,
    RISK_LABELS,
    RISK_ORDER,
    VALUE_LABELS,
    VALUE_ORDER,
    localize_segment_columns,
)




def render_segment_header() -> None:
    st.header("Сегменты")
    st.caption("Структура базы по ценности, риску и промо-зависимости для управленческой диагностики.")
    render_screen_help("segments")


def render_segment_filters_info(segment_user_base: pd.DataFrame) -> pd.DataFrame:
    localized = localize_segment_columns(segment_user_base)
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

    localized = localize_segment_columns(segment_map_table)
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

    localized = localize_segment_columns(segment_summary.copy()).sort_values(["users_count", "total_ltv_180d"], ascending=[False, False])
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
            "recommended_action_ru": "Rule-based интерпретация / возможное действие",
        }
    )

    display = table.drop(columns=["compound_segment", "Rule-based интерпретация / возможное действие"]).copy()
    for col in ["Доля пользователей", "Доля отмен", "Доля промо-поездок", "Средний response rate"]:
        display[col] = display[col].map(lambda x: format_percent(x, 1) if pd.notna(x) else "не рассчитывается в текущем срезе")
    for col in ["Средний LTV 180 дней", "Суммарный LTV 180 дней", "Средняя маржа завершённой поездки"]:
        display[col] = display[col].map(lambda x: format_currency(x, 0) if pd.notna(x) else "нет завершённых поездок")
    display["Средний recency"] = display["Средний recency"].map(lambda x: f"{format_number(x, 1)} дн." if pd.notna(x) else "нет завершённых поездок")
    display["Среднее число поездок за 90 дней"] = display["Среднее число поездок за 90 дней"].map(lambda x: format_number(x, 2) if pd.notna(x) else "недостаточно данных")

    st.dataframe(display, use_container_width=True)
    with st.expander("Интерпретация / возможное действие (rule-based)", expanded=False):
        st.caption("Этот блок — аналитическая эвристика на базе правил сегментации. Это не decision engine и не автоматический оркестратор действий.")
        actions_view = table[
            [
                "Комбинированный сегмент",
                "Пользователи",
                "Доля пользователей",
                "Средний LTV 180 дней",
                "Доля отмен",
                "Rule-based интерпретация / возможное действие",
            ]
        ].copy()
        actions_view["Доля пользователей"] = actions_view["Доля пользователей"].map(lambda x: format_percent(x, 1) if pd.notna(x) else "—")
        actions_view["Средний LTV 180 дней"] = actions_view["Средний LTV 180 дней"].map(lambda x: format_currency(x, 0) if pd.notna(x) else "—")
        actions_view["Доля отмен"] = actions_view["Доля отмен"].map(lambda x: format_percent(x, 1) if pd.notna(x) else "—")
        st.dataframe(actions_view, use_container_width=True)

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
    st.caption(f"Риск: {risk} · Ценность: {value} · Доминирующая промо-зависимость: {promo}")

    st.markdown("**Интерпретация / возможное действие (rule-based)**")
    action = ACTION_LABELS.get(selected_profile.get("recommended_action"), "Наблюдать без немедленного действия")
    st.info(
        f"{action}\n\n"
        "Пояснение: это аналитическая рекомендация по rule-based сегментации, а не decision engine "
        "и не предиктивная модель для автоматического принятия решений."
    )


def render_distribution_analytics(segment_user_base: pd.DataFrame) -> None:
    st.subheader("Распределительная аналитика")
    st.caption("Блоки ниже фокусируются на разбросе, хвостах и концентрации ценности, а не только на среднем уровне метрик.")

    chart_df = segment_user_base[["ltv_180d", "value_segment", "promo_trip_share", "recency_days"]].copy()
    chart_df["value_segment_ru"] = chart_df["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
    chart_df = chart_df.loc[chart_df["ltv_180d"].notna()]

    if chart_df.empty:
        st.info("Недостаточно данных для распределительных графиков.")
        return

    max_ltv = float(chart_df["ltv_180d"].quantile(0.99))
    max_ltv = max(max_ltv, 1.0)
    clipped_df = chart_df.assign(ltv_180d=chart_df["ltv_180d"].clip(lower=0, upper=max_ltv))

    t1, t2, t3 = st.tabs(["LTV и хвосты", "Поведение: recency и promo", "Концентрация ценности"])

    with t1:
        left, right = st.columns(2)
        with left:
            histogram = (
                alt.Chart(clipped_df)
                .mark_bar(opacity=0.8)
                .encode(
                    x=alt.X("ltv_180d:Q", bin=alt.Bin(maxbins=35), title="LTV 180 дней"),
                    y=alt.Y("count():Q", title="Пользователи"),
                    tooltip=[
                        alt.Tooltip("count():Q", title="Пользователи"),
                        alt.Tooltip("ltv_180d:Q", bin=True, title="Диапазон LTV"),
                    ],
                )
                .properties(height=280, title="Гистограмма LTV 180д (до 99-го перцентиля)")
            )
            st.altair_chart(histogram, use_container_width=True)
        with right:
            boxplot = (
                alt.Chart(clipped_df)
                .mark_boxplot(size=38)
                .encode(
                    x=alt.X("value_segment_ru:N", title="Сегмент ценности", sort=[VALUE_LABELS[x] for x in VALUE_ORDER]),
                    y=alt.Y("ltv_180d:Q", title="LTV 180 дней"),
                    color=alt.Color("value_segment_ru:N", title="Сегмент ценности", sort=[VALUE_LABELS[x] for x in VALUE_ORDER]),
                    tooltip=[
                        alt.Tooltip("value_segment_ru:N", title="Сегмент ценности"),
                        alt.Tooltip("median(ltv_180d):Q", title="Медиана LTV", format=",.0f"),
                    ],
                )
                .properties(height=280, title="Boxplot LTV 180д по сегментам ценности")
            )
            st.altair_chart(boxplot, use_container_width=True)

    with t2:
        recency_promo = chart_df[["recency_days", "promo_trip_share"]].copy()
        recency_promo = recency_promo.loc[recency_promo["recency_days"].notna() & recency_promo["promo_trip_share"].notna()]
        left, right = st.columns(2)
        with left:
            if recency_promo.empty:
                st.info("Недостаточно данных для распределения recency.")
            else:
                recency_max = float(recency_promo["recency_days"].quantile(0.99))
                recency_hist = (
                    alt.Chart(recency_promo.assign(recency_days=recency_promo["recency_days"].clip(lower=0, upper=max(recency_max, 1.0))))
                    .mark_bar(opacity=0.85)
                    .encode(
                        x=alt.X("recency_days:Q", bin=alt.Bin(maxbins=30), title="Recency, дней"),
                        y=alt.Y("count():Q", title="Пользователи"),
                        tooltip=[alt.Tooltip("count():Q", title="Пользователи"), alt.Tooltip("recency_days:Q", bin=True, title="Интервал recency")],
                    )
                    .properties(height=260, title="Распределение recency (до 99-го перцентиля)")
                )
                st.altair_chart(recency_hist, use_container_width=True)
        with right:
            if recency_promo.empty:
                st.info("Недостаточно данных для распределения promo_trip_share.")
            else:
                promo_hist = (
                    alt.Chart(recency_promo)
                    .mark_bar(opacity=0.85)
                    .encode(
                        x=alt.X("promo_trip_share:Q", bin=alt.Bin(maxbins=25), title="Доля промо-поездок"),
                        y=alt.Y("count():Q", title="Пользователи"),
                        tooltip=[alt.Tooltip("count():Q", title="Пользователи"), alt.Tooltip("promo_trip_share:Q", bin=True, title="Интервал доли промо")],
                    )
                    .properties(height=260, title="Распределение promo_trip_share")
                )
                st.altair_chart(promo_hist, use_container_width=True)

    with t3:
        concentration = get_ltv_concentration_by_value_segment(segment_user_base)
        concentration["value_segment_ru"] = concentration["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
        concentration = concentration.sort_values("total_ltv_180d", ascending=False)
        if concentration.empty:
            st.info("Недостаточно данных для концентрации ценности.")
        else:
            concentration["ltv_share_pct"] = concentration["ltv_share"] * 100
            concentration["users_share_pct"] = concentration["users_share"] * 100
            c_chart = (
                alt.Chart(concentration)
                .mark_bar()
                .encode(
                    x=alt.X("value_segment_ru:N", sort="-y", title="Сегмент ценности"),
                    y=alt.Y("ltv_share_pct:Q", title="Доля суммарного LTV, %"),
                    color=alt.Color("value_segment_ru:N", title="Сегмент ценности"),
                    tooltip=[
                        alt.Tooltip("value_segment_ru:N", title="Сегмент ценности"),
                        alt.Tooltip("users_count:Q", title="Пользователи", format=",.0f"),
                        alt.Tooltip("users_share_pct:Q", title="Доля пользователей", format=".1f"),
                        alt.Tooltip("ltv_share_pct:Q", title="Доля суммарного LTV", format=".1f"),
                        alt.Tooltip("total_ltv_180d:Q", title="Суммарный LTV 180д", format=",.0f"),
                    ],
                )
                .properties(height=260, title="Концентрация суммарного LTV по value-сегментам")
            )
            st.altair_chart(c_chart, use_container_width=True)
            view = concentration[["value_segment_ru", "users_count", "users_share", "ltv_share", "total_ltv_180d"]].rename(
                columns={
                    "value_segment_ru": "Сегмент ценности",
                    "users_count": "Пользователи",
                    "users_share": "Доля пользователей",
                    "ltv_share": "Доля суммарного LTV 180д",
                    "total_ltv_180d": "Суммарный LTV 180д",
                }
            )
            view["Доля пользователей"] = view["Доля пользователей"].map(lambda x: format_percent(x, 1) if pd.notna(x) else "—")
            view["Доля суммарного LTV 180д"] = view["Доля суммарного LTV 180д"].map(lambda x: format_percent(x, 1) if pd.notna(x) else "—")
            view["Суммарный LTV 180д"] = view["Суммарный LTV 180д"].map(lambda x: format_currency(x, 0) if pd.notna(x) else "—")
            st.dataframe(view, use_container_width=True)


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

    def _render_selected_vs_baseline_chart(df: pd.DataFrame, title: str, y_axis_title: str, value_format: str) -> None:
        if df.empty:
            st.info("Недостаточно данных для построения графика.")
            return
        long_df = df.melt(id_vars="metric", value_vars=["selected", "baseline"], var_name="profile", value_name="value")
        long_df["profile"] = long_df["profile"].map({"selected": "Выбранный сегмент", "baseline": "Эталон"})
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Метрика", sort=df["metric"].tolist()),
                xOffset=alt.XOffset("profile:N", sort=["Выбранный сегмент", "Эталон"]),
                y=alt.Y("value:Q", title=y_axis_title),
                color=alt.Color("profile:N", title="", sort=["Выбранный сегмент", "Эталон"]),
                tooltip=[
                    alt.Tooltip("metric:N", title="Метрика"),
                    alt.Tooltip("profile:N", title="Профиль"),
                    alt.Tooltip("value:Q", title="Значение", format=value_format),
                ],
            )
            .properties(height=260, title=title)
        )
        st.altair_chart(chart, use_container_width=True)

    if charts["monetary_metrics"].empty and charts["ratio_metrics"].empty:
        st.info("Недостаточно разнообразия сегментов для профильных графиков.")
    else:
        c_top1, c_top2 = st.columns(2)
        with c_top1:
            _render_selected_vs_baseline_chart(
                charts["monetary_metrics"],
                title="Денежный профиль: сегмент vs эталон",
                y_axis_title="Значение",
                value_format=",.2f",
            )
        with c_top2:
            _render_selected_vs_baseline_chart(
                charts["ratio_metrics"],
                title="Долевой профиль: сегмент vs эталон",
                y_axis_title="Значение, %",
                value_format=".2f",
            )

    unavailable_metrics = charts.get("unavailable_metrics", pd.DataFrame())
    if not unavailable_metrics.empty:
        reasons = (
            unavailable_metrics.groupby("reason")["metric"]
            .apply(lambda s: ", ".join(sorted(set(s))))
            .reset_index()
        )
        for _, row in reasons.iterrows():
            st.caption(f"{row['reason']}: {row['metric']}.")

    c1, c2 = st.columns(2)
    with c1:
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
    st.caption("Короткие наблюдения только по значимым отклонениям от эталона (медианы по сегментам), без шаблонных формулировок.")
    notes = generate_segment_diagnostics(selected_profile, baseline_profile)
    with st.expander("Показать диагностические наблюдения", expanded=False):
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
    render_distribution_analytics(filtered)
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
