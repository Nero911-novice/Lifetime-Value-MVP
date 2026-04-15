
from __future__ import annotations

import streamlit as st

from ..metrics import get_user_snapshot
from ..ui import (
    render_screen_help,
    render_methodology_footer,
    format_currency,
    format_number,
    format_percent,
    info_caption,
)


def _value(user, key, default=0):
    return user[key] if key in user.index else default


def render(data):
    st.header("Карточка пользователя")
    render_screen_help("user_profile")

    user_options = (
        data["user_mart"][["user_id", "home_city", "acquisition_channel", "activation_type"]]
        .assign(
            label=lambda d: d["user_id"] + " · " + d["home_city"] + " · " + d["acquisition_channel"]
        )
    )
    selected_label = st.selectbox(
        "Выберите пользователя",
        options=user_options["label"].tolist(),
        help="Карточка нужна для демонстрации связки между исходными событиями и агрегированной аналитикой.",
    )
    selected_user_id = user_options.loc[user_options["label"] == selected_label, "user_id"].iloc[0]
    snapshot = get_user_snapshot(selected_user_id, data)
    user = snapshot["user"]

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Исторический LTV 180д", format_currency(_value(user, "margin_180d", 0), 0))
    top2.metric("Созданные заказы", format_number(_value(user, "total_orders", 0), 0))
    top3.metric("Завершенные заказы", format_number(_value(user, "completed_orders", 0), 0))
    top4.metric("Доля отмен", format_percent(_value(user, "cancel_rate", 0), 1))

    mid1, mid2, mid3, mid4 = st.columns(4)
    mid1.metric("Response 7d", format_percent(_value(user, "response_rate_7d", 0), 1))
    mid2.metric("Активность 90д", "Да" if bool(_value(user, "active_90d_flag", False)) else "Нет")
    mid3.metric("Промо-зависимость", str(_value(user, "promo_band", "Неизвестно")))
    mid4.metric("Риск", str(_value(user, "risk_segment", "Не классифицирован")))

    st.subheader("Профиль пользователя")
    profile_cols = st.columns(3)
    profile_cols[0].markdown(
        f"""
**ID:** {user['user_id']}  
**Город:** {user['home_city']}  
**Канал:** {user['acquisition_channel']}  
**Тип активации:** {user['activation_type']}
"""
    )
    profile_cols[1].markdown(
        f"""
**Регистрация:** {user['registration_date'].date() if hasattr(user['registration_date'], 'date') and user['registration_date'] == user['registration_date'] else '—'}  
**Первая поездка:** {user['first_trip_date'].date() if hasattr(user['first_trip_date'], 'date') and user['first_trip_date'] == user['first_trip_date'] else '—'}  
**Последняя поездка:** {user['last_trip_ts'].date() if hasattr(user['last_trip_ts'], 'date') and user['last_trip_ts'] == user['last_trip_ts'] else '—'}  
**Статус демо:** {_value(user, 'lifecycle_status_demo', '—')}
"""
    )
    profile_cols[2].markdown(
        f"""
**Тариф:** {user['preferred_tariff']}  
**Подписка:** {user['subscription_plan']}  
**Ценность:** {_value(user, 'value_segment', 'Не классифицирован')}  
**Средняя маржа поездки:** {format_currency(_value(user, 'avg_trip_margin', 0), 0)}
"""
    )

    st.subheader("Событийный след пользователя")
    path1, path2, path3, path4 = st.columns(4)
    path1.metric("Касания", format_number(_value(user, "total_touches", 0), 0))
    path2.metric("Открытия", format_percent(_value(user, "open_rate", 0), 1))
    path3.metric("Клики", format_percent(_value(user, "click_rate", 0), 1))
    path4.metric("Конверсии 7д", format_percent(_value(user, "response_rate_7d", 0), 1))

    st.subheader("Последние поездки")
    info_caption(
        "На уровне заказов и поездок видно, из каких денежных элементов складывается contribution margin и как соотносятся созданные и завершённые заказы."
    )
    trip_cols = [
        "trip_id",
        "request_ts",
        "tariff",
        "order_status",
        "gmv",
        "platform_revenue",
        "promo_discount",
        "refund_amount",
        "variable_ops_cost",
        "contribution_margin",
    ]
    st.dataframe(snapshot["trips"][trip_cols].head(20), use_container_width=True)

    st.subheader("Последние маркетинговые касания")
    touch_cols = [
        "touch_id",
        "touch_ts",
        "touch_channel",
        "campaign_type",
        "offer_type",
        "opened_flag",
        "clicked_flag",
        "converted_within_7d_flag",
        "touch_cost",
    ]
    st.dataframe(snapshot["touches"][touch_cols].head(20), use_container_width=True)

    render_methodology_footer()
