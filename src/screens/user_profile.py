
from __future__ import annotations

import streamlit as st

from ..metrics import get_user_snapshot
from ..ui import render_screen_help, render_methodology_footer, format_currency, format_number, info_caption


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
    top1.metric("Исторический LTV 180д", format_currency(user["margin_180d"], 0))
    top2.metric("Поездки 365д", format_number(user["trips_365d"], 0))
    top3.metric("Response 7d", f'{user["response_rate_7d"]:.1%}')
    top4.metric("Риск", user["risk_segment"])

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
**Статус демо:** {user['lifecycle_status_demo']}
"""
    )
    profile_cols[2].markdown(
        f"""
**Тариф:** {user['preferred_tariff']}  
**Подписка:** {user['subscription_plan']}  
**Промо-зависимость:** {user['promo_band']}  
**Ценность:** {user['value_segment']}
"""
    )

    st.subheader("Последние поездки")
    info_caption("На уровне поездок видно, из каких денежных элементов складывается contribution margin пользователя.")
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
    st.dataframe(snapshot["trips"][trip_cols].head(15), use_container_width=True)

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
    st.dataframe(snapshot["touches"][touch_cols].head(15), use_container_width=True)

    render_methodology_footer()
