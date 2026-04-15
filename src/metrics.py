
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def apply_common_filters(
    user_mart: pd.DataFrame,
    city: str,
    channel: str,
    tariff: str,
    activation_type: str,
) -> pd.DataFrame:
    df = user_mart.copy()
    if city != "Все":
        df = df.loc[df["home_city"] == city]
    if channel != "Все":
        df = df.loc[df["acquisition_channel"] == channel]
    if tariff != "Все":
        df = df.loc[df["preferred_tariff"] == tariff]
    if activation_type != "Все":
        if activation_type == "Не активирован":
            df = df.loc[~df["activated_flag"]]
        else:
            df = df.loc[df["activation_type"] == activation_type]
    return df


def filter_related_tables(data: Dict[str, pd.DataFrame], filtered_users: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    user_ids = set(filtered_users["user_id"])
    return {
        "users": data["users"].loc[data["users"]["user_id"].isin(user_ids)].copy(),
        "trips": data["trips"].loc[data["trips"]["user_id"].isin(user_ids)].copy(),
        "marketing_touches": data["marketing_touches"].loc[data["marketing_touches"]["user_id"].isin(user_ids)].copy(),
        "campaigns": data["campaigns"].copy(),
        "data_dictionary": data["data_dictionary"].copy(),
        "dataset_summary": data["dataset_summary"].copy(),
    }




def _with_default_columns(df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of df with missing columns added using provided defaults."""
    if not defaults:
        return df
    result = df.copy()
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result

def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _get_observation_date(user_mart: pd.DataFrame, trips: pd.DataFrame | None = None) -> pd.Timestamp:
    if "observation_date" in user_mart.columns and len(user_mart):
        value = user_mart["observation_date"].iloc[0]
        if pd.notna(value):
            return pd.Timestamp(value).normalize()
    if trips is not None and len(trips):
        return pd.Timestamp(trips["request_ts"].max()).normalize()
    return pd.Timestamp.today().normalize()


def _period_order_metrics(trips: pd.DataFrame, end_date: pd.Timestamp, days: int = 30) -> dict:
    if trips.empty:
        return {
            "total_orders": 0,
            "completed_orders": 0,
            "cancelled_orders": 0,
            "cancel_rate": 0.0,
            "completed_margin": 0.0,
            "completed_gmv": 0.0,
        }
    start_date = end_date - pd.Timedelta(days=days - 1)
    window = trips.loc[(trips["request_ts"] >= start_date) & (trips["request_ts"] <= end_date)]
    completed = window.loc[window["order_status"] == "completed"]
    total_orders = int(len(window))
    completed_orders = int(len(completed))
    cancelled_orders = int(total_orders - completed_orders)
    return {
        "total_orders": total_orders,
        "completed_orders": completed_orders,
        "cancelled_orders": cancelled_orders,
        "cancel_rate": cancelled_orders / total_orders if total_orders else 0.0,
        "completed_margin": float(completed["contribution_margin"].sum()) if completed_orders else 0.0,
        "completed_gmv": float(completed["gmv"].sum()) if completed_orders else 0.0,
    }


def compute_overview_metrics(user_mart: pd.DataFrame, trips: pd.DataFrame | None = None) -> dict:
    trips = trips if trips is not None else pd.DataFrame(columns=["order_status", "request_ts", "contribution_margin", "gmv"])
    user_mart = _with_default_columns(
        user_mart,
        {
            "activated_flag": False,
            "active_90d_flag": False,
            "margin_180d": 0.0,
            "avg_trip_margin": 0.0,
            "acquisition_cost": 0.0,
            "completed_trips": 0.0,
        },
    )

    total_users = len(user_mart)
    activated_users = int(user_mart["activated_flag"].sum())
    total_orders = int(user_mart["total_orders"].sum()) if "total_orders" in user_mart else int(len(trips))
    completed_orders = int(user_mart["completed_orders"].sum()) if "completed_orders" in user_mart else int((trips["order_status"] == "completed").sum())
    completed_trips = int(user_mart["completed_trips"].sum()) if "completed_trips" in user_mart else completed_orders
    cancelled_orders = int(user_mart["cancelled_orders"].sum()) if "cancelled_orders" in user_mart else max(total_orders - completed_orders, 0)
    active_90d_share = float(user_mart["active_90d_flag"].mean()) if total_users else 0.0

    total_ltv_180 = float(user_mart["margin_180d"].sum()) if "margin_180d" in user_mart else 0.0
    ltv_180_mean = float(user_mart.loc[user_mart["activated_flag"], "margin_180d"].mean()) if activated_users else 0.0
    trip_margin_mean = float(user_mart.loc[user_mart["completed_trips"] > 0, "avg_trip_margin"].mean()) if completed_orders else 0.0

    total_cac = float(user_mart["acquisition_cost"].sum()) if "acquisition_cost" in user_mart else 0.0
    ltv_cac_ratio = total_ltv_180 / total_cac if total_cac > 0 else np.nan
    cancel_rate = cancelled_orders / total_orders if total_orders else 0.0

    observation_date = _get_observation_date(user_mart, trips)
    current_period = _period_order_metrics(trips, observation_date, days=30)
    previous_period = _period_order_metrics(trips, observation_date - pd.Timedelta(days=30), days=30)

    current_activation_start = observation_date - pd.Timedelta(days=29)
    previous_activation_start = observation_date - pd.Timedelta(days=59)
    current_new_activations = int(
        user_mart["first_trip_date"].between(current_activation_start, observation_date, inclusive="both").sum()
    ) if "first_trip_date" in user_mart else 0
    previous_new_activations = int(
        user_mart["first_trip_date"].between(previous_activation_start, observation_date - pd.Timedelta(days=30), inclusive="both").sum()
    ) if "first_trip_date" in user_mart else 0

    current_new_regs = int(
        user_mart["registration_date"].between(current_activation_start, observation_date, inclusive="both").sum()
    ) if "registration_date" in user_mart else 0
    previous_new_regs = int(
        user_mart["registration_date"].between(previous_activation_start, observation_date - pd.Timedelta(days=30), inclusive="both").sum()
    ) if "registration_date" in user_mart else 0

    return {
        "total_users": total_users,
        "activated_users": activated_users,
        "activation_rate": activated_users / total_users if total_users else 0.0,
        "total_orders": total_orders,
        "completed_orders": completed_orders,
        "completed_trips": completed_trips,
        "cancelled_orders": cancelled_orders,
        "cancel_rate": cancel_rate,
        "active_90d_share": active_90d_share,
        "ltv_180_mean": ltv_180_mean,
        "total_ltv_180": total_ltv_180,
        "avg_trip_margin": trip_margin_mean,
        "ltv_cac_ratio": ltv_cac_ratio,
        "current_period": current_period,
        "previous_period": previous_period,
        "current_new_activations": current_new_activations,
        "previous_new_activations": previous_new_activations,
        "current_new_registrations": current_new_regs,
        "previous_new_registrations": previous_new_regs,
    }


def build_overview_charts(user_mart: pd.DataFrame, trips: pd.DataFrame) -> dict:
    user_mart = _with_default_columns(
        user_mart,
        {
            "activated_flag": False,
            "cohort_month": "Не активирован",
            "margin_180d": 0.0,
            "active_90d_flag": False,
            "cancel_rate": 0.0,
            "registration_date": pd.NaT,
            "risk_segment": "Не классифицирован",
            "value_segment": "Не классифицирован",
            "acquisition_channel": "Неизвестно",
            "acquisition_cost": 0.0,
            "home_city": "Неизвестно",
            "promo_band": "Неизвестно",
        },
    )

    activated = user_mart.loc[user_mart["activated_flag"]].copy()
    cohort_trend = (
        activated.groupby("cohort_month", dropna=False)
        .agg(
            activated_users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
        )
        .reset_index()
        .sort_values("cohort_month")
    )

    registrations_trend = (
        user_mart.assign(registration_month=user_mart["registration_date"].dt.to_period("M").astype("string"))
        .groupby("registration_month", dropna=False)
        .agg(
            registered_users=("user_id", "count"),
            activated_users=("activated_flag", "sum"),
        )
        .reset_index()
        .sort_values("registration_month")
    )

    risk_value_map = (
        user_mart.groupby(["risk_segment", "value_segment"], dropna=False)
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
        )
        .reset_index()
    )

    risk_value_pivot = risk_value_map.pivot_table(
        index="risk_segment",
        columns="value_segment",
        values="users",
        fill_value=0,
    )

    channel_summary = (
        user_mart.groupby("acquisition_channel")
        .agg(
            users=("user_id", "count"),
            activation_rate=("activated_flag", "mean"),
            avg_ltv_180=("margin_180d", "mean"),
            cancel_rate=("cancel_rate", "mean"),
            avg_cac=("acquisition_cost", "mean"),
        )
        .reset_index()
        .sort_values("avg_ltv_180", ascending=False)
    )

    city_summary = (
        user_mart.groupby("home_city")
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
            cancel_rate=("cancel_rate", "mean"),
        )
        .reset_index()
        .sort_values("avg_ltv_180", ascending=False)
    )

    segment_action_map = (
        user_mart.groupby(["risk_segment", "value_segment", "promo_band"], dropna=False)
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
        )
        .reset_index()
    )
    segment_action_map["recommended_action"] = segment_action_map.apply(_recommend_action, axis=1)
    segment_action_map = segment_action_map.sort_values(["users", "avg_ltv_180"], ascending=[False, False]).head(12)

    completed = trips.loc[trips["order_status"] == "completed"].copy()
    if not completed.empty:
        monthly_ops = (
            completed.assign(month=completed["request_ts"].dt.to_period("M").astype("string"))
            .groupby("month")
            .agg(
                completed_orders=("trip_id", "count"),
                total_margin=("contribution_margin", "sum"),
                total_gmv=("gmv", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )
    else:
        monthly_ops = pd.DataFrame(columns=["month", "completed_orders", "total_margin", "total_gmv"])

    return {
        "cohort_trend": cohort_trend,
        "registrations_trend": registrations_trend,
        "risk_value_map": risk_value_map,
        "risk_value_pivot": risk_value_pivot,
        "channel_summary": channel_summary,
        "city_summary": city_summary,
        "segment_action_map": segment_action_map,
        "monthly_ops": monthly_ops,
    }


def build_key_changes_table(metrics: dict) -> pd.DataFrame:
    current = metrics["current_period"]
    previous = metrics["previous_period"]

    def delta(cur: float, prev: float) -> float:
        return cur - prev

    rows = [
        {
            "Показатель": "Созданные заказы за 30д",
            "Текущий период": current["total_orders"],
            "Предыдущий период": previous["total_orders"],
            "Изменение": delta(current["total_orders"], previous["total_orders"]),
            "Интерпретация": "Изменение общего объема спроса в рамках выбранного среза.",
        },
        {
            "Показатель": "Завершенные заказы за 30д",
            "Текущий период": current["completed_orders"],
            "Предыдущий период": previous["completed_orders"],
            "Изменение": delta(current["completed_orders"], previous["completed_orders"]),
            "Интерпретация": "Динамика реально состоявшихся поездок, а не только созданного спроса.",
        },
        {
            "Показатель": "Доля отмен за 30д",
            "Текущий период": current["cancel_rate"],
            "Предыдущий период": previous["cancel_rate"],
            "Изменение": delta(current["cancel_rate"], previous["cancel_rate"]),
            "Интерпретация": "Рост отмен может указывать на проблемы предложения, цены или качества сервиса.",
        },
        {
            "Показатель": "Маржа завершенных поездок за 30д",
            "Текущий период": current["completed_margin"],
            "Предыдущий период": previous["completed_margin"],
            "Изменение": delta(current["completed_margin"], previous["completed_margin"]),
            "Интерпретация": "Это ближайший к управленческой экономике слой среди оперативных показателей.",
        },
        {
            "Показатель": "Новые активации за 30д",
            "Текущий период": metrics["current_new_activations"],
            "Предыдущий период": metrics["previous_new_activations"],
            "Изменение": delta(metrics["current_new_activations"], metrics["previous_new_activations"]),
            "Интерпретация": "Показывает, как меняется скорость перевода регистраций в первую завершенную поездку.",
        },
        {
            "Показатель": "Новые регистрации за 30д",
            "Текущий период": metrics["current_new_registrations"],
            "Предыдущий период": metrics["previous_new_registrations"],
            "Изменение": delta(metrics["current_new_registrations"], metrics["previous_new_registrations"]),
            "Интерпретация": "Нужен для чтения верхней воронки, отдельно от активации и удержания.",
        },
    ]
    return pd.DataFrame(rows)


def compute_cohort_matrices(user_mart: pd.DataFrame, trips: pd.DataFrame, max_age_months: int = 12) -> dict:
    user_mart = _with_default_columns(
        user_mart,
        {
            "activated_flag": False,
            "first_trip_date": pd.NaT,
            "cohort_month": "Не активирован",
            "margin_180d": 0.0,
            "active_90d_flag": False,
            "total_orders": 0.0,
            "completed_orders": 0.0,
            "cancel_rate": 0.0,
            "acquisition_cost": 0.0,
        },
    )

    activated = user_mart.loc[user_mart["activated_flag"], ["user_id", "first_trip_date", "cohort_month"]].copy()
    cohort_sizes = activated.groupby("cohort_month")["user_id"].nunique()

    completed = trips.loc[trips["order_status"] == "completed"].copy()
    completed = completed.merge(activated, on="user_id", how="inner")
    completed["age_month"] = (
        (completed["request_ts"].dt.year - completed["first_trip_date"].dt.year) * 12
        + (completed["request_ts"].dt.month - completed["first_trip_date"].dt.month)
    )
    completed = completed.loc[completed["age_month"].between(0, max_age_months)]

    retention = (
        completed.groupby(["cohort_month", "age_month"])["user_id"]
        .nunique()
        .div(cohort_sizes, level=0)
        .unstack(fill_value=0.0)
        .sort_index()
    )

    monthly_margin = (
        completed.groupby(["cohort_month", "age_month"])["contribution_margin"]
        .sum()
        .div(cohort_sizes, level=0)
        .unstack(fill_value=0.0)
        .sort_index()
    )
    cumulative_margin = monthly_margin.cumsum(axis=1)

    observation_period = pd.Period(_get_observation_date(user_mart, trips), freq="M")
    maturity = (
        activated.groupby("cohort_month")["user_id"]
        .count()
        .rename("activated_users")
        .to_frame()
    )
    maturity["maturity_months"] = [
        (observation_period.year - pd.Period(c, freq="M").year) * 12
        + (observation_period.month - pd.Period(c, freq="M").month)
        for c in maturity.index
    ]

    cohort_summary = (
        user_mart.loc[user_mart["activated_flag"]]
        .groupby("cohort_month")
        .agg(
            activated_users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
            avg_orders=("total_orders", "mean"),
            avg_completed_orders=("completed_orders", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
            avg_cac=("acquisition_cost", "mean"),
        )
        .reset_index()
        .merge(maturity.reset_index(), on=["cohort_month", "activated_users"], how="left")
        .sort_values("cohort_month")
    )

    retention.columns = [f"M{int(col)}" for col in retention.columns]
    cumulative_margin.columns = [f"M{int(col)}" for col in cumulative_margin.columns]

    return {
        "retention_matrix": retention,
        "cohort_ltv_matrix": cumulative_margin,
        "cohort_maturity": maturity.reset_index().sort_values("cohort_month"),
        "cohort_summary": cohort_summary,
        "monthly_margin": monthly_margin,
    }


def build_selected_cohort_curves(matrices: dict, cohort_month: str) -> pd.DataFrame:
    retention = matrices["retention_matrix"]
    ltv = matrices["cohort_ltv_matrix"]
    if cohort_month not in retention.index or cohort_month not in ltv.index:
        return pd.DataFrame(columns=["age_month", "retention", "cumulative_ltv"])
    retention_row = retention.loc[cohort_month]
    ltv_row = ltv.loc[cohort_month]
    age_months = [int(col.replace("M", "")) for col in retention_row.index]
    return pd.DataFrame(
        {
            "age_month": age_months,
            "retention": retention_row.values,
            "cumulative_ltv": ltv_row.values,
        }
    )


def build_segment_table(user_mart: pd.DataFrame) -> pd.DataFrame:
    user_mart = _with_default_columns(
        user_mart,
        {
            "risk_segment": "Не классифицирован",
            "value_segment": "Не классифицирован",
            "promo_band": "Неизвестно",
            "margin_180d": 0.0,
            "recent_trips_90d": 0.0,
            "response_rate_7d": 0.0,
            "active_90d_flag": False,
            "cancel_rate": 0.0,
        },
    )

    segment = (
        user_mart.groupby(["risk_segment", "value_segment", "promo_band"], dropna=False)
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            avg_trips_90d=("recent_trips_90d", "mean"),
            avg_response_7d=("response_rate_7d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
        )
        .reset_index()
        .sort_values(["users", "avg_ltv_180"], ascending=[False, False])
    )
    segment["recommended_action"] = segment.apply(_recommend_action, axis=1)
    return segment


def _recommend_action(row: pd.Series) -> str:
    risk = row.get("risk_segment")
    value = row.get("value_segment")
    promo = row.get("promo_band")
    if value == "Высокая ценность" and risk == "Низкий риск":
        return "Защитный режим: не субсидировать, поддерживать качество"
    if value == "Высокая ценность" and risk in {"Средний риск", "Высокий риск"}:
        return "Приоритетное удержание: мягкий стимул и контроль качества"
    if value == "Высокая ценность" and risk == "Спящий":
        return "Реактивационный тест: персональное возвращение"
    if value == "Средняя ценность" and promo == "Высокая":
        return "Ограничить дорогие промо, тестировать точечные офферы"
    if value == "Низкая ценность" and promo == "Высокая":
        return "Сократить субсидии, оставить дешевые касания"
    if risk == "Не активирован":
        return "Дожим до первой поездки: активационный сценарий"
    return "Наблюдение и стандартная коммуникация"


def build_risk_distribution(user_mart: pd.DataFrame) -> pd.DataFrame:
    return (
        user_mart.groupby("risk_segment")
        .agg(users=("user_id", "count"), avg_ltv_180=("margin_180d", "mean"))
        .reset_index()
        .sort_values("users", ascending=False)
    )


def build_value_distribution(user_mart: pd.DataFrame) -> pd.DataFrame:
    return (
        user_mart.groupby("value_segment")
        .agg(users=("user_id", "count"), avg_ltv_180=("margin_180d", "mean"))
        .reset_index()
        .sort_values("users", ascending=False)
    )


def build_risk_value_pivot(user_mart: pd.DataFrame, metric: str = "users") -> pd.DataFrame:
    user_mart = _with_default_columns(
        user_mart,
        {
            "risk_segment": "Не классифицирован",
            "value_segment": "Не классифицирован",
            "margin_180d": 0.0,
            "cancel_rate": 0.0,
            "active_90d_flag": False,
        },
    )

    source = (
        user_mart.groupby(["risk_segment", "value_segment"], dropna=False)
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            avg_cancel_rate=("cancel_rate", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
        )
        .reset_index()
    )
    return source.pivot_table(index="risk_segment", columns="value_segment", values=metric, fill_value=0)


def get_user_snapshot(user_id: str, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame | pd.Series]:
    user_row = data["user_mart"].loc[data["user_mart"]["user_id"] == user_id].iloc[0]
    trips = (
        data["trips"]
        .loc[data["trips"]["user_id"] == user_id]
        .sort_values("request_ts", ascending=False)
        .copy()
    )
    touches = (
        data["marketing_touches"]
        .loc[data["marketing_touches"]["user_id"] == user_id]
        .sort_values("touch_ts", ascending=False)
        .copy()
    )
    return {"user": user_row, "trips": trips, "touches": touches}


def build_data_model_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = data["dataset_summary"].copy()
    result["role_in_demo"] = [
        "Пользовательский контекст и источник привлечения",
        "Факт заказов, завершений, отмен, оборота и маржи",
        "История коммуникаций и response-подход",
        "Агрегация кампаний и тестовых воздействий",
        "Документация полей",
    ]
    return result


def build_screen_data_dependency_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Экран": "Обзор",
                "Ключевые таблицы": "users, trips, marketing_touches",
                "Что демонстрирует": "Быструю управленческую ориентацию: LTV, CAC, заказы, отмены, ключевые изменения",
            },
            {
                "Экран": "Когорты",
                "Ключевые таблицы": "users, trips",
                "Что демонстрирует": "Качество новых когорт, накопление LTV, зрелость и retention по месяцам жизни",
            },
            {
                "Экран": "Сегменты",
                "Ключевые таблицы": "user_mart, marketing_touches",
                "Что демонстрирует": "Связь риска, ценности и промо-зависимости с потенциальным действием",
            },
            {
                "Экран": "Карточка пользователя",
                "Ключевые таблицы": "users, trips, marketing_touches",
                "Что демонстрирует": "Переход от исходных событий к агрегатам и управленческой интерпретации",
            },
            {
                "Экран": "Модель данных",
                "Ключевые таблицы": "all",
                "Что демонстрирует": "Структуру будущей БД и аналитическую роль каждого слоя данных",
            },
        ]
    )
