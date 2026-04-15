
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

COHORT_MAX_HORIZON = 18


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


def _resolve_user_columns(users_df: pd.DataFrame) -> pd.DataFrame:
    users = users_df.copy()
    rename_map = {
        "home_city": "city",
        "first_trip_date": "first_completed_trip_date",
        "acquisition_cost": "cac",
    }
    users = users.rename(columns=rename_map)
    defaults = {
        "city": "Неизвестно",
        "activation_type": "Неизвестно",
        "preferred_tariff": "Неизвестно",
        "acquisition_channel": "Неизвестно",
        "cac": 0.0,
        "first_completed_trip_date": pd.NaT,
        "registration_date": pd.NaT,
    }
    return _with_default_columns(users, defaults)


def _resolve_trip_columns(trips_df: pd.DataFrame) -> pd.DataFrame:
    trips = trips_df.copy()
    rename_map = {
        "request_ts": "order_created_at",
        "order_status": "trip_status",
        "gmv": "gross_booking",
        "variable_ops_cost": "variable_cost",
    }
    trips = trips.rename(columns=rename_map)
    trips = _with_default_columns(
        trips,
        {
            "order_created_at": pd.NaT,
            "trip_status": "unknown",
            "promo_discount": 0.0,
            "driver_bonus": 0.0,
            "refund_amount": 0.0,
            "variable_cost": 0.0,
            "platform_revenue": 0.0,
            "contribution_margin": 0.0,
            "gross_booking": 0.0,
        },
    )
    if "is_completed" not in trips.columns:
        trips["is_completed"] = trips["trip_status"].eq("completed")
    if "is_cancelled" not in trips.columns:
        trips["is_cancelled"] = ~trips["is_completed"]
    if "is_promo_trip" not in trips.columns:
        trips["is_promo_trip"] = trips["promo_discount"] > 0
    return trips


def _lifecycle_month(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["order_created_at"].dt.year - frame["first_completed_trip_date"].dt.year) * 12
        + (frame["order_created_at"].dt.month - frame["first_completed_trip_date"].dt.month)
    )


def _observation_date_from_trips(trips_df: pd.DataFrame) -> pd.Timestamp:
    if trips_df.empty:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(trips_df["order_created_at"].max()).normalize()


def build_cohort_user_base(users_df: pd.DataFrame, trips_df: pd.DataFrame, touches_df: pd.DataFrame) -> pd.DataFrame:
    del touches_df  # explicitly kept for interface extensibility
    users = _resolve_user_columns(users_df)
    trips = _resolve_trip_columns(trips_df)

    activated_users = users.loc[users["first_completed_trip_date"].notna()].copy()
    if activated_users.empty:
        return pd.DataFrame()

    activated_users["cohort_month_period"] = activated_users["first_completed_trip_date"].dt.to_period("M")
    activated_users["cohort_month"] = activated_users["cohort_month_period"].astype("string")
    activated_users["cohort_label"] = activated_users["cohort_month"]
    activated_users["activation_month"] = activated_users["first_completed_trip_date"].dt.to_period("M").astype("string")

    observation_date = _observation_date_from_trips(trips)
    observation_period = observation_date.to_period("M")
    activated_users["maturity_months"] = (
        (observation_period.year - activated_users["cohort_month_period"].dt.year) * 12
        + (observation_period.month - activated_users["cohort_month_period"].dt.month)
    ).clip(lower=0)
    cohort_sizes = activated_users.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")
    activated_users = activated_users.merge(cohort_sizes, on="cohort_month", how="left")

    user_trips = trips.merge(
        activated_users[["user_id", "first_completed_trip_date"]],
        on="user_id",
        how="inner",
    )
    user_trips["months_since_first_trip"] = _lifecycle_month(user_trips)
    user_trips = user_trips.loc[user_trips["months_since_first_trip"] >= 0].copy()
    user_trips["days_since_first_trip"] = (
        user_trips["order_created_at"].dt.normalize() - user_trips["first_completed_trip_date"].dt.normalize()
    ).dt.days

    agg_orders = user_trips.groupby("user_id").agg(
        created_orders_count=("trip_id", "count"),
        completed_orders_count=("is_completed", "sum"),
        cancelled_orders_count=("is_cancelled", "sum"),
    )
    agg_orders["cancellation_rate"] = np.where(
        agg_orders["created_orders_count"] > 0,
        agg_orders["cancelled_orders_count"] / agg_orders["created_orders_count"],
        0.0,
    )

    completed = user_trips.loc[user_trips["is_completed"]].copy()
    promo_share = completed.groupby("user_id").agg(
        promo_trip_share=("is_promo_trip", "mean"),
        refund_trip_share=("refund_amount", lambda s: float((s > 0).mean())),
    )
    rides_last_30 = completed.loc[completed["order_created_at"] >= observation_date - pd.Timedelta(days=30)].groupby("user_id").size().rename("rides_last_30d")
    rides_last_90 = completed.loc[completed["order_created_at"] >= observation_date - pd.Timedelta(days=90)].groupby("user_id").size().rename("rides_last_90d")

    ltv_parts = {}
    for horizon in (30, 90, 180, 365):
        cutoff = completed.loc[completed["days_since_first_trip"].between(0, horizon, inclusive="both")]
        ltv_parts[f"ltv_{horizon}d"] = cutoff.groupby("user_id")["contribution_margin"].sum()
    ltv_df = pd.DataFrame(ltv_parts).fillna(0.0)

    month_flags = {}
    max_horizon = min(COHORT_MAX_HORIZON, int(user_trips["months_since_first_trip"].max()) if not user_trips.empty else 0)
    for month in range(max_horizon + 1):
        active = (
            user_trips.loc[(user_trips["is_completed"]) & (user_trips["months_since_first_trip"] == month)]
            .groupby("user_id")
            .size()
            .rename(f"is_active_month_{month}")
            .gt(0)
            .astype(int)
        )
        month_flags[f"is_active_month_{month}"] = active
    flags_df = pd.DataFrame(month_flags).fillna(0).astype(int) if month_flags else pd.DataFrame(index=activated_users["user_id"])

    result = activated_users[
        [
            "user_id",
            "cohort_month",
            "cohort_label",
            "cohort_size",
            "first_completed_trip_date",
            "activation_month",
            "city",
            "acquisition_channel",
            "activation_type",
            "preferred_tariff",
            "maturity_months",
        ]
    ].copy()
    for frame in (agg_orders, promo_share, rides_last_30, rides_last_90, ltv_df, flags_df):
        result = result.merge(frame, on="user_id", how="left")

    fill_zero_columns = [
        "created_orders_count",
        "completed_orders_count",
        "cancelled_orders_count",
        "cancellation_rate",
        "promo_trip_share",
        "refund_trip_share",
        "rides_last_30d",
        "rides_last_90d",
        "ltv_30d",
        "ltv_90d",
        "ltv_180d",
        "ltv_365d",
    ] + [col for col in result.columns if col.startswith("is_active_month_")]
    for col in fill_zero_columns:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    result["months_since_first_trip"] = result["maturity_months"]
    return result


def get_cohort_summary(cohort_user_base: pd.DataFrame) -> pd.DataFrame:
    if cohort_user_base.empty:
        return pd.DataFrame(columns=["cohort_month", "cohort_size"])
    summary = (
        cohort_user_base.groupby("cohort_month", as_index=False)
        .agg(
            cohort_size=("user_id", "nunique"),
            avg_ltv_30d=("ltv_30d", "mean"),
            avg_ltv_90d=("ltv_90d", "mean"),
            avg_ltv_180d=("ltv_180d", "mean"),
            avg_margin_per_completed_order=("ltv_365d", "sum"),
            activation_rate=("user_id", "size"),
            avg_cancellation_rate=("cancellation_rate", "mean"),
            avg_promo_trip_share=("promo_trip_share", "mean"),
            avg_completed_orders_90d=("rides_last_90d", "mean"),
            avg_created_orders_90d=("created_orders_count", "mean"),
            maturity_months=("maturity_months", "max"),
        )
        .sort_values("cohort_month")
    )
    summary["activation_rate"] = 1.0
    summary["avg_margin_per_completed_order"] = np.where(
        summary["avg_completed_orders_90d"] > 0,
        summary["avg_ltv_180d"] / summary["avg_completed_orders_90d"].clip(lower=1e-9),
        0.0,
    )
    for month in (1, 3, 6):
        col = f"is_active_month_{month}"
        summary[f"retention_m{month}"] = (
            cohort_user_base.groupby("cohort_month")[col].mean().reindex(summary["cohort_month"]).fillna(np.nan).values
            if col in cohort_user_base.columns
            else np.nan
        )
    return summary


def get_cohort_maturity_table(cohort_user_base: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    if cohort_user_base.empty:
        return pd.DataFrame(columns=["cohort_month", "cohort_size", "maturity_months"])
    max_period = pd.Timestamp(max_date).to_period("M")
    maturity = (
        cohort_user_base.groupby("cohort_month")["user_id"].nunique().rename("cohort_size").to_frame()
    )
    periods = pd.PeriodIndex(maturity.index, freq="M")
    maturity_months = (max_period.year - periods.year) * 12 + (max_period.month - periods.month)
    maturity["maturity_months"] = np.maximum(maturity_months, 0).astype(int)
    return maturity.reset_index().sort_values("cohort_month")


def _build_metric_matrix(
    cohort_user_base: pd.DataFrame,
    trips_df: pd.DataFrame,
    metric: str,
    max_horizon: int = COHORT_MAX_HORIZON,
) -> pd.DataFrame:
    if cohort_user_base.empty:
        return pd.DataFrame()
    trips = _resolve_trip_columns(trips_df)
    if "first_completed_trip_date" not in cohort_user_base.columns:
        return pd.DataFrame()
    merged = trips.merge(
        cohort_user_base[["user_id", "cohort_month", "first_completed_trip_date"]],
        on="user_id",
        how="inner",
    )
    merged["month_index"] = _lifecycle_month(merged)
    merged = merged.loc[merged["month_index"].between(0, max_horizon)].copy()
    cohort_size = cohort_user_base.groupby("cohort_month")["user_id"].nunique()

    if metric == "retention":
        base = (
            merged.loc[merged["is_completed"]]
            .groupby(["cohort_month", "month_index"])["user_id"]
            .nunique()
            .div(cohort_size, level=0)
        )
    elif metric == "cum_ltv":
        base = (
            merged.loc[merged["is_completed"]]
            .groupby(["cohort_month", "month_index"])["contribution_margin"]
            .sum()
            .div(cohort_size, level=0)
        ).groupby(level=0).cumsum()
    elif metric == "cum_margin":
        base = (
            merged.loc[merged["is_completed"]]
            .groupby(["cohort_month", "month_index"])["contribution_margin"]
            .sum()
            .div(cohort_size, level=0)
        ).groupby(level=0).cumsum()
    elif metric == "cancellation":
        created = merged.groupby(["cohort_month", "month_index"])["trip_id"].count()
        cancelled = merged.groupby(["cohort_month", "month_index"])["is_cancelled"].sum()
        base = (cancelled / created.replace({0: np.nan}))
    elif metric == "promo_share":
        completed = merged.loc[merged["is_completed"]]
        denom = completed.groupby(["cohort_month", "month_index"])["trip_id"].count()
        num = completed.groupby(["cohort_month", "month_index"])["is_promo_trip"].sum()
        base = (num / denom.replace({0: np.nan}))
    elif metric == "rides_per_user":
        base = (
            merged.loc[merged["is_completed"]]
            .groupby(["cohort_month", "month_index"])["trip_id"]
            .count()
            .div(cohort_size, level=0)
        ).groupby(level=0).cumsum()
    else:
        raise ValueError(f"Unknown metric mode: {metric}")

    matrix = base.unstack("month_index").sort_index()
    maturity = cohort_user_base.groupby("cohort_month")["maturity_months"].max()
    for month in matrix.columns:
        matrix.loc[maturity < month, month] = np.nan
    matrix.columns = [f"M{int(c)}" for c in matrix.columns]
    return matrix


def build_retention_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="retention")


def build_cumulative_ltv_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="cum_ltv")


def build_cumulative_margin_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="cum_margin")


def build_cancellation_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="cancellation")


def build_promo_share_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="promo_share")


def build_rides_per_user_matrix(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    return _build_metric_matrix(cohort_user_base, trips_df, metric="rides_per_user")


def build_cohort_size_matrix(cohort_summary: pd.DataFrame) -> pd.DataFrame:
    if cohort_summary.empty:
        return pd.DataFrame()
    size_matrix = cohort_summary.set_index("cohort_month")[["cohort_size"]]
    size_matrix.columns = ["M0"]
    return size_matrix


def get_selected_cohort_profile(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame, cohort_month: str) -> dict:
    summary = get_cohort_summary(cohort_user_base)
    row = summary.loc[summary["cohort_month"] == cohort_month]
    if row.empty:
        return {}
    profile = row.iloc[0].to_dict()
    selected_users = cohort_user_base.loc[cohort_user_base["cohort_month"] == cohort_month]
    profile["cohort_month"] = cohort_month
    profile["rides_per_user_90d"] = float(selected_users["rides_last_90d"].mean()) if len(selected_users) else 0.0
    profile["trips_df_rows"] = len(trips_df.loc[trips_df["user_id"].isin(selected_users["user_id"])]) if len(selected_users) else 0
    return profile


def get_selected_cohort_curves(cohort_user_base: pd.DataFrame, trips_df: pd.DataFrame, cohort_month: str) -> dict:
    matrices = {
        "retention": build_retention_matrix(cohort_user_base, trips_df),
        "ltv": build_cumulative_ltv_matrix(cohort_user_base, trips_df),
        "margin": build_cumulative_margin_matrix(cohort_user_base, trips_df),
        "rides": build_rides_per_user_matrix(cohort_user_base, trips_df),
    }
    curves = {}
    for key, matrix in matrices.items():
        if matrix.empty or cohort_month not in matrix.index:
            curves[key] = pd.DataFrame(columns=["month_index", "selected", "baseline"])
            continue
        baseline = matrix.median(axis=0, skipna=True)
        selected = matrix.loc[cohort_month]
        curves[key] = pd.DataFrame(
            {
                "month_index": [int(col.replace("M", "")) for col in selected.index],
                "selected": selected.values,
                "baseline": baseline.values,
            }
        )
    return curves


def compare_cohort_to_baseline(
    cohort_summary: pd.DataFrame,
    selected_cohort: str,
    baseline_mode: str = "median",
) -> pd.DataFrame:
    if cohort_summary.empty or selected_cohort not in set(cohort_summary["cohort_month"]):
        return pd.DataFrame(columns=["metric", "selected", "baseline", "delta"])
    metric_map = {
        "Размер": "cohort_size",
        "Зрелость": "maturity_months",
        "Retention M1": "retention_m1",
        "Retention M3": "retention_m3",
        "Retention M6": "retention_m6",
        "LTV 30д": "avg_ltv_30d",
        "LTV 90д": "avg_ltv_90d",
        "LTV 180д": "avg_ltv_180d",
        "Средняя маржа поездки": "avg_margin_per_completed_order",
        "Доля отмен": "avg_cancellation_rate",
        "Доля промо": "avg_promo_trip_share",
        "Поездки на пользователя 90д": "avg_completed_orders_90d",
    }
    selected_row = cohort_summary.loc[cohort_summary["cohort_month"] == selected_cohort].iloc[0]
    baseline_row = cohort_summary.median(numeric_only=True) if baseline_mode == "median" else cohort_summary.mean(numeric_only=True)
    rows = []
    for label, col in metric_map.items():
        sel = float(selected_row[col]) if col in selected_row and pd.notna(selected_row[col]) else np.nan
        base = float(baseline_row[col]) if col in baseline_row and pd.notna(baseline_row[col]) else np.nan
        rows.append({"metric": label, "selected": sel, "baseline": base, "delta": sel - base})
    return pd.DataFrame(rows)


def generate_cohort_diagnostics(selected_profile: dict, baseline_profile: dict) -> list[str]:
    notes: list[str] = []
    if not selected_profile or not baseline_profile:
        return ["Недостаточно данных для диагностики: выберите когорту с достаточной зрелостью и размером."]

    if selected_profile.get("retention_m1", np.nan) > baseline_profile.get("retention_m1", np.nan) and selected_profile.get("avg_ltv_180d", np.nan) < baseline_profile.get("avg_ltv_180d", np.nan):
        notes.append("Retention M1 выше медианы, но LTV 180д ниже — это может указывать на лучшее удержание при более слабой монетизации.")
    if selected_profile.get("avg_ltv_180d", np.nan) > baseline_profile.get("avg_ltv_180d", np.nan) and selected_profile.get("avg_promo_trip_share", np.nan) > baseline_profile.get("avg_promo_trip_share", np.nan):
        notes.append("Когорта даёт высокий LTV, но с повышенной долей промо-поездок — может быть связано с ценой такого роста и требует проверки юнит-экономики.")
    if selected_profile.get("avg_cancellation_rate", np.nan) > baseline_profile.get("avg_cancellation_rate", np.nan):
        notes.append("Доля отмен выше эталона — это может указывать на операционные ограничения или нестабильный клиентский опыт в этом срезе.")
    if selected_profile.get("retention_m1", np.nan) < baseline_profile.get("retention_m1", np.nan) and selected_profile.get("avg_completed_orders_90d", np.nan) < baseline_profile.get("avg_completed_orders_90d", np.nan):
        notes.append("Слабость проявляется уже в раннем цикле: ниже retention M1 и ниже поездок на пользователя за 90 дней.")
    if selected_profile.get("cohort_size", 0) < baseline_profile.get("cohort_size", 0):
        notes.append("Размер когорты ниже медианы, поэтому выводы по отклонениям стоит интерпретировать осторожно из-за более высокой волатильности.")
    return notes[:5] if notes else ["Когорта близка к медианному эталону по ключевым метрикам; заметных диагностических отклонений не выявлено."]


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
