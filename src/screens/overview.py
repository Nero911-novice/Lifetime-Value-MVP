
from __future__ import annotations

import streamlit as st

from ..metrics import compute_overview_metrics, build_overview_charts, build_key_changes_table
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


def render(user_mart, trips):
    st.header("Обзор")
    render_screen_help("overview")

    metrics = compute_overview_metrics(user_mart, trips)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Пользователи", format_number(metrics["total_users"], 0))
    top2.metric("Активация", format_percent(metrics["activation_rate"], 1))
    top3.metric("Суммарный LTV 180д", format_currency(metrics["total_ltv_180"], 0))
    top4.metric("LTV/CAC", "—" if metrics["ltv_cac_ratio"] != metrics["ltv_cac_ratio"] else f'{metrics["ltv_cac_ratio"]:.2f}')

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
    st.dataframe(key_changes, use_container_width=True)
    render_item_help("key_changes", "Пояснение к ключевым изменениям")

    render_methodology_footer()
