
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


def _fmt_date(value) -> str:
    if value is None:
        return "—"
    if hasattr(value, "date") and value == value:
        return str(value.date())
    return "—"


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

    st.subheader("Путь пользователя: активация → ценность")
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("LTV 180д", format_currency(_value(user, "ltv_180d", _value(user, "margin_180d", 0)), 0))
    top2.metric("LTV 365д", format_currency(_value(user, "ltv_365d", _value(user, "margin_365d", 0)), 0))
    top3.metric("Созданные заказы", format_number(_value(user, "created_orders_count", _value(user, "total_orders", 0)), 0))
    top4.metric("Завершённые заказы", format_number(_value(user, "completed_orders_count", _value(user, "completed_orders", 0)), 0))

    mid1, mid2, mid3, mid4, mid5 = st.columns(5)
    mid1.metric("Доля отмен", format_percent(_value(user, "cancellation_rate", _value(user, "cancel_rate", 0)), 1))
    mid2.metric("Promo trip share", format_percent(_value(user, "promo_trip_share", 0), 1))
    mid3.metric("Refund share", format_percent(_value(user, "refund_trip_share", _value(user, "refund_rate", 0)), 1))
    mid4.metric("Rides 30/90", f"{format_number(_value(user, 'rides_last_30d', 0), 0)} / {format_number(_value(user, 'rides_last_90d', 0), 0)}")
    mid5.metric("Recency", f"{format_number(_value(user, 'recency_days', 0), 0)} дн.")

    st.subheader("Профиль пользователя")
    profile_cols = st.columns(4)
    profile_cols[0].markdown(
        f"""
**ID:** {user['user_id']}  
**Город:** {_value(user, 'home_city', _value(user, 'city', '—'))}  
**Канал привлечения:** {_value(user, 'acquisition_channel', '—')}  
**Тип активации:** {_value(user, 'activation_type', '—')}
"""
    )
    profile_cols[1].markdown(
        f"""
**Тариф:** {_value(user, 'preferred_tariff', '—')}  
**Подписка:** {_value(user, 'subscription_plan', '—')}  
**Risk segment:** {_value(user, 'risk_segment', '—')}  
**Value segment:** {_value(user, 'value_segment', '—')}
"""
    )
    profile_cols[2].markdown(
        f"""
**Регистрация:** {_fmt_date(_value(user, 'registration_date', None))}  
**Первая завершённая поездка:** {_fmt_date(_value(user, 'first_completed_trip_date', _value(user, 'first_trip_date', None)))}  
**Tenure:** {format_number(_value(user, 'tenure_days', 0), 0)} дн.  
**Response 7d:** {format_percent(_value(user, 'responded_7d_rate', _value(user, 'response_rate_7d', 0)), 1)}
"""
    )
    profile_cols[3].markdown(
        f"""
**Promo dependency:** {_value(user, 'promo_dependency_segment', _value(user, 'promo_band', '—'))}  
**Recommended action:** {_value(user, 'recommended_action', '—')}  
**Total contribution margin:** {format_currency(_value(user, 'total_contribution_margin', _value(user, 'total_margin', 0)), 0)}  
**Avg margin / completed order:** {format_currency(_value(user, 'avg_margin_per_completed_order', _value(user, 'avg_trip_margin', 0)), 0)}
"""
    )

    st.subheader("Событийный след и маркетинг")
    path1, path2, path3, path4, path5 = st.columns(5)
    path1.metric("Касания", format_number(_value(user, "total_touches", 0), 0))
    path2.metric("Открытия", format_percent(_value(user, "open_rate", 0), 1))
    path3.metric("Клики", format_percent(_value(user, "click_rate", 0), 1))
    path4.metric("Конверсии 7д", format_percent(_value(user, "responded_7d_rate", _value(user, "response_rate_7d", 0)), 1))
    path5.metric("Активен 90д", "Да" if bool(_value(user, "is_active_90d", _value(user, "active_90d_flag", False))) else "Нет")

    st.subheader("Почему пользователь в текущем сегменте")
    expl_cols = st.columns(3)
    explainability = snapshot.get("explainability", {})
    with expl_cols[0]:
        st.markdown("**Risk segment**")
        for reason in explainability.get("risk_reasons", []):
            st.caption(f"• {reason}")
    with expl_cols[1]:
        st.markdown("**Value segment**")
        for reason in explainability.get("value_reasons", []):
            st.caption(f"• {reason}")
    with expl_cols[2]:
        st.markdown("**Promo dependency**")
        for reason in explainability.get("promo_reasons", []):
            st.caption(f"• {reason}")

    st.subheader("Краткая интерпретация")
    for note in snapshot.get("interpretation", []):
        st.caption(f"• {note}")

    st.subheader("Хронология касаний и поездок")
    info_caption("Единая временная лента связывает коммуникации, заказы и экономику пользователя в одном контексте.")
    timeline = snapshot.get("timeline")
    if timeline is not None and len(timeline):
        st.dataframe(timeline, use_container_width=True, hide_index=True)
    else:
        st.info("Недостаточно событий для построения временной ленты.")

    st.subheader("Последние поездки (детализация)")
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

    st.subheader("Последние маркетинговые касания (детализация)")
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
