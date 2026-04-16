
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..metrics import (
    build_key_changes_table,
    build_overview_charts,
    build_overview_next_steps,
    compute_overview_metrics,
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
from ..segment_labels import localize_segment_columns


def _delta_text(current: float, previous: float, is_percent: bool = False) -> str:
    delta = current - previous
    if is_percent:
        return ("+" if delta >= 0 else "") + format_percent(delta, 1)
    return ("+" if delta >= 0 else "") + format_number(delta, 0)


def _format_change_row(row: pd.Series) -> tuple[str, str, str]:
    unit = row["unit"]
    if unit == "rate":
        current = format_percent(row["Текущий период"], 1)
        previous = format_percent(row["Предыдущий период"], 1)
        delta_value = ("+" if row["Изменение"] >= 0 else "") + format_percent(row["Изменение"], 1)
    elif unit == "currency":
        current = format_currency(row["Текущий период"], 0)
        previous = format_currency(row["Предыдущий период"], 0)
        delta_value = ("+" if row["Изменение"] >= 0 else "") + format_currency(row["Изменение"], 0)
    else:
        current = format_number(row["Текущий период"], 0)
        previous = format_number(row["Предыдущий период"], 0)
        delta_value = ("+" if row["Изменение"] >= 0 else "") + format_number(row["Изменение"], 0)
    return current, previous, delta_value


def render(user_mart, trips):
    st.header("Обзор")
    render_screen_help("overview")

    metrics = compute_overview_metrics(user_mart, trips)

    st.subheader("Управленческий срез")
    info_caption("Сверху — размер и экономика базы, ниже — операционный контур и динамика пополнения за последние 30 дней.")

    with st.container(border=True):
        st.caption("Размер базы и экономика")
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Пользователи", format_number(metrics["total_users"], 0))
        top2.metric("Активация", format_percent(metrics["activation_rate"], 1))
        top3.metric("Суммарный LTV 180д", format_currency(metrics["total_ltv_180"], 0))
        top4.metric("LTV/CAC", "—" if metrics["ltv_cac_ratio"] != metrics["ltv_cac_ratio"] else f'{metrics["ltv_cac_ratio"]:.2f}')

    with st.container(border=True):
        st.caption("Операционный контур (текущий 30-дневный срез)")
        ops1, ops2, ops3, ops4 = st.columns(4)
        ops1.metric(
            "Созданные заказы",
            format_number(metrics["total_orders"], 0),
            delta=_delta_text(metrics["current_period"]["total_orders"], metrics["previous_period"]["total_orders"]),
        )
        ops2.metric(
            "Завершенные заказы",
            format_number(metrics["completed_orders"], 0),
            delta=_delta_text(metrics["current_period"]["completed_orders"], metrics["previous_period"]["completed_orders"]),
        )
        ops3.metric(
            "Доля отмен",
            format_percent(metrics["cancel_rate"], 1),
            delta=_delta_text(metrics["current_period"]["cancel_rate"], metrics["previous_period"]["cancel_rate"], is_percent=True),
            delta_color="inverse",
        )
        ops4.metric("Активные за 90 дней", format_percent(metrics["active_90d_share"], 1))

    with st.container(border=True):
        st.caption("Качество новых пользователей (текущие 30 дней vs предыдущие 30 дней)")
        val1, val2, val3, val4 = st.columns(4)
        val1.metric("Средний LTV 180д на активированного", format_currency(metrics["ltv_180_mean"], 0))
        val2.metric("Средняя маржа поездки", format_currency(metrics["avg_trip_margin"], 0))
        val3.metric(
            "Новые активации за 30д",
            format_number(metrics["current_new_activations"], 0),
            delta=_delta_text(metrics["current_new_activations"], metrics["previous_new_activations"]),
        )
        val4.metric(
            "Новые регистрации за 30д",
            format_number(metrics["current_new_registrations"], 0),
            delta=_delta_text(metrics["current_new_registrations"], metrics["previous_new_registrations"]),
        )

    render_item_help("total_ltv_180", "Пояснение к суммарному LTV 180д")
    render_item_help("hist_ltv_180", "Пояснение к среднему LTV 180д")
    render_item_help("total_orders", "Пояснение к созданным заказам")
    render_item_help("cancel_rate", "Пояснение к доле отмен")
    render_item_help("contribution_margin", "Пояснение к contribution margin")
    render_item_help("ltv_cac_ratio", "Пояснение к LTV/CAC")

    charts = build_overview_charts(user_mart, trips)

    st.subheader("Динамика новых когорт и регистраций")
    info_caption(
        "Здесь соединены две логики: верхняя воронка регистрации и нижняя воронка активации в первую завершенную поездку."
    )
    cohort_col, reg_col = st.columns(2)
    with cohort_col:
        st.caption("Когорты активации")
        st.line_chart(
            charts["cohort_trend"].set_index("cohort_month")[["activated_users", "avg_ltv_180"]],
            height=320,
        )
    with reg_col:
        st.caption("Регистрации и последующая активация")
        st.line_chart(
            charts["registrations_trend"].set_index("registration_month")[["registered_users", "activated_users"]],
            height=320,
        )

    st.subheader("Карта сегментов риска и ценности")
    info_caption(
        "Матрица показывает, где в выбранном срезе сосредоточена база. Это управленческий переход от общих KPI к приоритизации сегментов."
    )
    st.dataframe(
        charts["risk_value_pivot"].style.background_gradient(cmap="Blues").format("{:,.0f}"),
        use_container_width=True,
    )
    render_item_help("risk_value_map", "Пояснение к карте риска и ценности")

    st.subheader("Ключевые сегменты для действия")
    info_caption(
        "Ниже показаны самые крупные сочетания риска, ценности и промо-зависимости с rule-based рекомендацией. Это демонстрационный мост к будущему decision layer."
    )
    segment_action_map = localize_segment_columns(charts["segment_action_map"]).rename(
        columns={
            "risk_segment_ru": "Риск",
            "value_segment_ru": "Ценность",
            "promo_dependency_segment_ru": "Промо-зависимость",
            "users_count": "Пользователи",
            "users_share": "Доля пользователей",
            "avg_ltv_180d": "Средний LTV 180 дней",
            "avg_cancellation_rate": "Средняя доля отмен",
            "recommended_action_ru": "Рекомендованное действие",
        }
    )
    st.dataframe(
        segment_action_map[[
            "Риск",
            "Ценность",
            "Промо-зависимость",
            "Пользователи",
            "Доля пользователей",
            "Средний LTV 180 дней",
            "Средняя доля отмен",
            "Рекомендованное действие",
        ]],
        use_container_width=True,
    )

    st.subheader("Сравнение каналов привлечения")
    info_caption(
        "Канал нужен не только для активации, но и для оценки качества базы: сколько пользователей привлечено, как они активируются, какой дают LTV и какова их средняя доля отмен."
    )
    st.dataframe(
        charts["channel_summary"].rename(
            columns={
                "acquisition_channel": "Канал",
                "users": "Пользователи",
                "activation_rate": "Активация",
                "avg_ltv_180": "Средний LTV 180д",
                "cancel_rate": "Доля отмен",
                "avg_cac": "Средний CAC",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Городской срез")
    city_left, city_right = st.columns(2)
    with city_left:
        st.caption("Города по LTV и активности")
        st.bar_chart(
            charts["city_summary"].set_index("home_city")[["avg_ltv_180", "active_90d_share"]],
            height=300,
        )
    with city_right:
        st.caption("Операционная динамика по месяцам")
        st.line_chart(
            charts["monthly_ops"].set_index("month")[["completed_orders", "total_margin"]],
            height=300,
        )

    st.subheader("Ключевые изменения за период")
    info_caption(
        "Сравнение текущих и предыдущих 30 дней помогает увидеть, меняется ли реальный операционный и экономический контур, а не только статичный исторический срез."
    )
    key_changes = build_key_changes_table(metrics)
    top_changes = key_changes.head(4)
    change_cols = st.columns(len(top_changes))
    for idx, (_, row) in enumerate(top_changes.iterrows()):
        current_value, previous_value, delta_value = _format_change_row(row)
        with change_cols[idx]:
            with st.container(border=True):
                st.caption(row["Показатель"])
                st.write(f"**{current_value}**")
                st.caption(f"Было: {previous_value}")
                st.caption(f"Δ: {delta_value}")
                st.caption(row["Интерпретация"])
    with st.expander("Полный список изменений (30д к 30д)", expanded=False):
        st.dataframe(
            key_changes[["Показатель", "Текущий период", "Предыдущий период", "Изменение", "Изменение %", "Интерпретация"]],
            use_container_width=True,
        )
    render_item_help("key_changes", "Пояснение к ключевым изменениям")

    st.subheader("Куда смотреть дальше")
    info_caption("Короткая навигация по следующим экранам на основе наблюдаемой динамики. Это rule-based подсказка, а не decision engine.")
    next_steps = build_overview_next_steps(metrics)
    for step in next_steps:
        with st.container(border=True):
            st.markdown(f"**→ {step['screen']}** · {step['focus']}")
            st.caption(step["why"])

    st.subheader("Быстрые переходы по детализации")
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.caption("Сегменты с наибольшей долей пользователей")
        top_segments_preview = segment_action_map[[
            "Риск",
            "Ценность",
            "Промо-зависимость",
            "Пользователи",
            "Доля пользователей",
            "Рекомендованное действие",
        ]].head(5)
        st.dataframe(top_segments_preview, use_container_width=True, height=220)
        st.caption("Для полной структуры и сравнения baseline откройте экран Segments.")
    with preview_right:
        st.caption("Каналы с максимальной базой пользователей")
        channel_preview = charts["channel_summary"].rename(
            columns={
                "acquisition_channel": "Канал",
                "users": "Пользователи",
                "activation_rate": "Активация",
                "avg_ltv_180": "Средний LTV 180д",
            }
        )[["Канал", "Пользователи", "Активация", "Средний LTV 180д"]].head(5)
        st.dataframe(channel_preview, use_container_width=True, height=220)
        st.caption("Для диагностики по месяцам активации перейдите в Cohorts.")

    render_methodology_footer()
