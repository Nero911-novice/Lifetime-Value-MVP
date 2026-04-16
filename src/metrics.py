
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .segment_labels import RISK_LABELS, RISK_ORDER, VALUE_LABELS, VALUE_ORDER

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
            "acquisition_channel": "Неизвестно",
            "acquisition_cost": 0.0,
            "home_city": "Неизвестно",
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

    segment_user_base = build_segment_user_base(user_mart, trips, touches_df=None)
    risk_value_map = get_segment_map_table(segment_user_base)

    risk_value_pivot = (
        risk_value_map.assign(
            risk_segment_ru=lambda d: d["risk_segment"].map(RISK_LABELS),
            value_segment_ru=lambda d: d["value_segment"].map(VALUE_LABELS),
        )
        .pivot_table(
            index="risk_segment_ru",
            columns="value_segment_ru",
            values="users_count",
            fill_value=0,
            aggfunc="sum",
        )
        .reindex(index=[RISK_LABELS[x] for x in RISK_ORDER], columns=[VALUE_LABELS[x] for x in VALUE_ORDER], fill_value=0)
    )

    segment_action_map = (
        segment_user_base.groupby(["risk_segment", "value_segment", "promo_dependency_segment"], dropna=False)
        .agg(
            users_count=("user_id", "nunique"),
            avg_ltv_180d=("ltv_180d", "mean"),
            created_orders_total=("created_orders_count", "sum"),
            cancelled_orders_total=("cancelled_orders_count", "sum"),
            recommended_action=("recommended_action", lambda s: s.mode().iloc[0] if not s.mode().empty else "Observe / No immediate action"),
        )
        .reset_index()
    )
    total_segment_users = max(len(segment_user_base), 1)
    segment_action_map["users_share"] = segment_action_map["users_count"] / total_segment_users
    segment_action_map["avg_cancellation_rate"] = segment_action_map["cancelled_orders_total"] / segment_action_map["created_orders_total"].replace({0: np.nan})
    segment_action_map = segment_action_map.sort_values(["users_count", "avg_ltv_180d"], ascending=[False, False]).head(12)

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
    agg_orders["cancellation_rate"] = agg_orders["cancelled_orders_count"] / agg_orders["created_orders_count"].replace({0: np.nan})

    completed = user_trips.loc[user_trips["is_completed"]].copy()
    promo_share = completed.groupby("user_id").agg(
        promo_trip_share=("is_promo_trip", "mean"),
        refund_trip_share=("refund_amount", lambda s: float((s > 0).mean())),
    )
    rides_last_30 = completed.loc[completed["order_created_at"] >= observation_date - pd.Timedelta(days=30)].groupby("user_id").size().rename("rides_last_30d")
    rides_last_90 = completed.loc[completed["order_created_at"] >= observation_date - pd.Timedelta(days=90)].groupby("user_id").size().rename("rides_last_90d")
    created_last_90 = user_trips.loc[user_trips["order_created_at"] >= observation_date - pd.Timedelta(days=90)].groupby("user_id").size().rename("created_orders_90d")
    cancelled_last_90 = user_trips.loc[
        (user_trips["order_created_at"] >= observation_date - pd.Timedelta(days=90)) & user_trips["is_cancelled"]
    ].groupby("user_id").size().rename("cancelled_orders_90d")

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
    for frame in (agg_orders, promo_share, rides_last_30, rides_last_90, created_last_90, cancelled_last_90, ltv_df, flags_df):
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
        "created_orders_90d",
        "cancelled_orders_90d",
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
            total_ltv_90d=("ltv_90d", "sum"),
            activation_rate=("user_id", "size"),
            total_cancelled_orders_90d=("cancelled_orders_90d", "sum"),
            avg_promo_trip_share=("promo_trip_share", "mean"),
            avg_completed_orders_90d=("rides_last_90d", "mean"),
            total_completed_orders_90d=("rides_last_90d", "sum"),
            total_created_orders_90d=("created_orders_90d", "sum"),
            avg_created_orders_90d=("created_orders_90d", "mean"),
            maturity_months=("maturity_months", "max"),
        )
        .sort_values("cohort_month")
    )
    summary["activation_rate"] = 1.0
    summary["avg_margin_per_completed_order"] = np.where(
        summary["total_completed_orders_90d"] > 0,
        summary["total_ltv_90d"] / summary["total_completed_orders_90d"],
        np.nan,
    )
    summary["avg_cancellation_rate"] = np.where(
        summary["total_created_orders_90d"] > 0,
        summary["total_cancelled_orders_90d"] / summary["total_created_orders_90d"],
        np.nan,
    )
    summary = summary.drop(columns=["total_ltv_90d", "total_completed_orders_90d", "total_created_orders_90d", "total_cancelled_orders_90d"])
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


def build_segment_user_base(user_mart_df: pd.DataFrame, trips_df: pd.DataFrame | None = None, touches_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build per-user segment mart with value/risk/promo labels and recommended actions."""
    segment_base = _with_default_columns(
        user_mart_df.copy(),
        {
            "home_city": "Неизвестно",
            "city": "Неизвестно",
            "acquisition_channel": "Неизвестно",
            "activation_type": "Неизвестно",
            "preferred_tariff": "Неизвестно",
            "registration_date": pd.NaT,
            "first_trip_date": pd.NaT,
            "total_orders": 0,
            "created_orders_count": 0,
            "completed_orders": 0,
            "completed_orders_count": 0,
            "cancelled_orders": 0,
            "cancelled_orders_count": 0,
            "cancel_rate": np.nan,
            "margin_30d": 0.0,
            "margin_90d": 0.0,
            "margin_180d": 0.0,
            "margin_365d": 0.0,
            "total_margin": 0.0,
            "avg_trip_margin": np.nan,
            "recent_trips_30d": 0.0,
            "recent_trips_90d": 0.0,
            "recency_days": np.nan,
            "promo_trip_share": np.nan,
            "refund_rate": np.nan,
            "response_rate_7d": np.nan,
            "converted_touches_7d": 0.0,
            "total_touches": 0.0,
            "acquisition_cost": 0.0,
            "cac": 0.0,
            "active_90d_flag": False,
        },
    )

    if "city" in user_mart_df.columns:
        city_source = user_mart_df["city"]
    else:
        city_source = segment_base["home_city"]
    segment_base["city"] = city_source.where(city_source.notna() & (city_source.astype(str) != "Неизвестно"), segment_base["home_city"])
    segment_base["first_completed_trip_date"] = pd.to_datetime(segment_base.get("first_trip_date"), errors="coerce")
    segment_base["registration_date"] = pd.to_datetime(segment_base.get("registration_date"), errors="coerce")

    observation_date = _get_observation_date(segment_base, trips_df)
    segment_base["observation_date"] = observation_date
    segment_base["tenure_days"] = (observation_date - segment_base["registration_date"]).dt.days

    source_created = pd.to_numeric(segment_base.get("total_orders"), errors="coerce").fillna(0)
    source_completed = pd.to_numeric(segment_base.get("completed_orders"), errors="coerce").fillna(0)
    source_cancelled = pd.to_numeric(segment_base.get("cancelled_orders"), errors="coerce").fillna(0)
    current_created = pd.to_numeric(segment_base.get("created_orders_count"), errors="coerce")
    current_completed = pd.to_numeric(segment_base.get("completed_orders_count"), errors="coerce")
    current_cancelled = pd.to_numeric(segment_base.get("cancelled_orders_count"), errors="coerce")

    segment_base["created_orders_count"] = current_created.where((current_created > 0) | (source_created <= 0), source_created).fillna(source_created)
    segment_base["completed_orders_count"] = current_completed.where((current_completed > 0) | (source_completed <= 0), source_completed).fillna(source_completed)
    segment_base["cancelled_orders_count"] = current_cancelled.where((current_cancelled > 0) | (source_cancelled <= 0), source_cancelled).fillna(source_cancelled)

    ltv_30_source = segment_base["ltv_30d"] if "ltv_30d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    ltv_90_source = segment_base["ltv_90d"] if "ltv_90d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    ltv_180_source = segment_base["ltv_180d"] if "ltv_180d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    ltv_365_source = segment_base["ltv_365d"] if "ltv_365d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    total_margin_source = segment_base["total_contribution_margin"] if "total_contribution_margin" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    rides_30_source = segment_base["rides_last_30d"] if "rides_last_30d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    rides_90_source = segment_base["rides_last_90d"] if "rides_last_90d" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    refund_source = segment_base["refund_trip_share"] if "refund_trip_share" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)
    responded_source = segment_base["responded_7d_rate"] if "responded_7d_rate" in segment_base.columns else pd.Series(np.nan, index=segment_base.index)

    segment_base["ltv_30d"] = pd.to_numeric(ltv_30_source, errors="coerce").fillna(pd.to_numeric(segment_base["margin_30d"], errors="coerce")).fillna(0)
    segment_base["ltv_90d"] = pd.to_numeric(ltv_90_source, errors="coerce").fillna(pd.to_numeric(segment_base["margin_90d"], errors="coerce")).fillna(0)
    segment_base["ltv_180d"] = pd.to_numeric(ltv_180_source, errors="coerce").fillna(pd.to_numeric(segment_base["margin_180d"], errors="coerce")).fillna(0)
    segment_base["ltv_365d"] = pd.to_numeric(ltv_365_source, errors="coerce").fillna(pd.to_numeric(segment_base["margin_365d"], errors="coerce")).fillna(0)
    segment_base["total_contribution_margin"] = pd.to_numeric(total_margin_source, errors="coerce").fillna(pd.to_numeric(segment_base["total_margin"], errors="coerce")).fillna(0)
    segment_base["rides_last_30d"] = pd.to_numeric(rides_30_source, errors="coerce").fillna(pd.to_numeric(segment_base["recent_trips_30d"], errors="coerce")).fillna(0)
    segment_base["rides_last_90d"] = pd.to_numeric(rides_90_source, errors="coerce").fillna(pd.to_numeric(segment_base["recent_trips_90d"], errors="coerce")).fillna(0)
    segment_base["refund_trip_share"] = pd.to_numeric(refund_source, errors="coerce").fillna(pd.to_numeric(segment_base["refund_rate"], errors="coerce")).fillna(0)
    segment_base["responded_7d_rate"] = pd.to_numeric(responded_source, errors="coerce").fillna(pd.to_numeric(segment_base["response_rate_7d"], errors="coerce"))
    segment_base["cac"] = segment_base.get("cac", segment_base["acquisition_cost"])
    segment_base["is_active_90d"] = segment_base.get("is_active_90d", segment_base["active_90d_flag"])

    if touches_df is not None and not touches_df.empty:
        touches = touches_df.copy()
        touch_agg = touches.groupby("user_id").agg(
            touches_count=("touch_id", "count"),
            converted_count=("converted_within_7d_flag", "sum"),
        )
        touch_agg["responded_7d_rate"] = touch_agg["converted_count"] / touch_agg["touches_count"].replace({0: np.nan})
        segment_base = segment_base.merge(touch_agg[["responded_7d_rate"]], on="user_id", how="left", suffixes=("", "_touch"))
        segment_base["responded_7d_rate"] = segment_base["responded_7d_rate_touch"].combine_first(segment_base["responded_7d_rate"])
        segment_base = segment_base.drop(columns=["responded_7d_rate_touch"])

    segment_base["avg_margin_per_completed_order"] = np.where(
        segment_base["completed_orders_count"] > 0,
        segment_base["total_contribution_margin"] / segment_base["completed_orders_count"].replace({0: np.nan}),
        np.nan,
    )
    segment_base["cancellation_rate"] = np.where(
        segment_base["created_orders_count"] > 0,
        segment_base["cancelled_orders_count"] / segment_base["created_orders_count"].replace({0: np.nan}),
        np.nan,
    )
    segment_base["promo_trip_share"] = pd.to_numeric(segment_base["promo_trip_share"], errors="coerce")
    segment_base["responded_7d_rate"] = pd.to_numeric(segment_base["responded_7d_rate"], errors="coerce")
    segment_base["recency_days"] = pd.to_numeric(segment_base["recency_days"], errors="coerce")

    segment_base["value_segment"] = assign_value_segment(segment_base)
    segment_base["risk_segment"] = assign_risk_segment(segment_base)
    segment_base["promo_dependency_segment"] = assign_promo_dependency_segment(segment_base)
    segment_base["compound_segment"] = assign_compound_segment(segment_base)
    segment_base["recommended_action"] = assign_recommended_action(segment_base)

    expected_cols = [
        "user_id", "city", "acquisition_channel", "activation_type", "preferred_tariff",
        "registration_date", "first_completed_trip_date", "tenure_days", "created_orders_count",
        "completed_orders_count", "cancelled_orders_count", "cancellation_rate", "ltv_30d", "ltv_90d",
        "ltv_180d", "ltv_365d", "total_contribution_margin", "avg_margin_per_completed_order",
        "rides_last_30d", "rides_last_90d", "recency_days", "promo_trip_share", "refund_trip_share",
        "responded_7d_rate", "value_segment", "risk_segment", "promo_dependency_segment",
        "compound_segment", "recommended_action", "is_active_90d", "cac",
    ]
    for col in expected_cols:
        if col not in segment_base.columns:
            segment_base[col] = np.nan

    return segment_base[expected_cols].copy()


def assign_value_segment(df: pd.DataFrame) -> pd.Series:
    ltv = pd.to_numeric(df.get("ltv_180d"), errors="coerce")
    completed_orders = pd.to_numeric(df.get("completed_orders_count"), errors="coerce").fillna(0)
    margin = pd.to_numeric(df.get("total_contribution_margin"), errors="coerce").fillna(0)

    active_mask = completed_orders > 0
    valid_ltv = ltv[active_mask & ltv.notna()]
    if len(valid_ltv) >= 20:
        q1, q2 = valid_ltv.quantile([0.35, 0.70]).tolist()
    else:
        q1, q2 = 120.0, 420.0

    segment = pd.Series("Medium value", index=df.index, dtype="string")
    segment = segment.mask((ltv < q1) | ((ltv <= 0) & active_mask), "Low value")
    segment = segment.mask(ltv >= q2, "High value")
    segment = segment.mask((completed_orders == 0) | (margin <= 0), "Low value")
    return segment.fillna("Low value")


def assign_risk_segment(df: pd.DataFrame) -> pd.Series:
    recency = pd.to_numeric(df.get("recency_days"), errors="coerce")
    rides_30 = pd.to_numeric(df.get("rides_last_30d"), errors="coerce").fillna(0)
    rides_90 = pd.to_numeric(df.get("rides_last_90d"), errors="coerce").fillna(0)
    completed_orders = pd.to_numeric(df.get("completed_orders_count"), errors="coerce").fillna(0)

    stable_mask = ((recency <= 14) & (rides_30 >= 1)) | ((recency <= 30) & (rides_90 >= 4))
    cooling_mask = ((recency > 14) & (recency <= 45)) | ((recency <= 35) & (rides_90.between(1, 3)))
    at_risk_mask = ((recency > 45) & (recency <= 90)) | ((rides_90 <= 1) & (recency > 35))
    dormant_mask = (recency > 90) | ((rides_90 == 0) & (completed_orders > 0))
    no_completed_mask = completed_orders == 0

    risk = np.select(
        [no_completed_mask | dormant_mask, stable_mask, at_risk_mask, cooling_mask],
        ["Dormant", "Stable / Active", "At risk", "Cooling"],
        default="Cooling",
    )
    return pd.Series(risk, index=df.index, dtype="string").fillna("Cooling")


def assign_promo_dependency_segment(df: pd.DataFrame) -> pd.Series:
    promo_share = pd.to_numeric(df.get("promo_trip_share"), errors="coerce")
    response_rate = pd.to_numeric(df.get("responded_7d_rate"), errors="coerce")

    promo_component = promo_share.fillna(0).clip(0, 1)
    response_component = response_rate.fillna(response_rate.median(skipna=True) if response_rate.notna().any() else 0).clip(0, 1)
    score = 0.7 * promo_component + 0.3 * response_component

    segment = pd.cut(
        score,
        bins=[-0.01, 0.25, 0.55, 1.01],
        labels=["Low promo dependency", "Medium promo dependency", "High promo dependency"],
    )
    return segment.astype("string").fillna("Low promo dependency")


def assign_compound_segment(df: pd.DataFrame) -> pd.Series:
    return (df["risk_segment"].astype("string") + " | " + df["value_segment"].astype("string")).astype("string")


def assign_recommended_action(df: pd.DataFrame) -> pd.Series:
    actions = []
    for _, row in df.iterrows():
        value = row.get("value_segment")
        risk = row.get("risk_segment")
        promo = row.get("promo_dependency_segment")

        if value == "High value" and risk in {"At risk", "Dormant"}:
            action = "Protect / Retain" if risk == "At risk" else "Reactivate"
        elif value == "High value" and risk == "Cooling":
            action = "Protect / Retain"
        elif value == "High value" and risk == "Stable / Active":
            action = "Observe / No immediate action"
        elif value == "Medium value" and risk == "Cooling" and promo in {"Medium promo dependency", "High promo dependency"}:
            action = "Stimulate carefully"
        elif value == "Low value" and promo == "High promo dependency":
            action = "Limit incentives"
        elif value == "Low value" and risk == "Dormant":
            action = "Observe / No immediate action"
        elif risk in {"At risk", "Dormant"}:
            action = "Reactivate"
        else:
            action = "Observe / No immediate action"
        actions.append(action)
    return pd.Series(actions, index=df.index, dtype="string")


def get_segment_summary(segment_user_base: pd.DataFrame) -> pd.DataFrame:
    if segment_user_base.empty:
        return pd.DataFrame(columns=["compound_segment", "risk_segment", "value_segment"])

    total_users = len(segment_user_base)
    summary = (
        segment_user_base.groupby(["compound_segment", "risk_segment", "value_segment"], dropna=False)
        .agg(
            users_count=("user_id", "nunique"),
            avg_ltv_180d=("ltv_180d", "mean"),
            median_ltv_180d=("ltv_180d", "median"),
            avg_margin_per_completed_order=("avg_margin_per_completed_order", "mean"),
            avg_created_orders=("created_orders_count", "mean"),
            avg_completed_orders=("completed_orders_count", "mean"),
            created_orders_total=("created_orders_count", "sum"),
            cancelled_orders_total=("cancelled_orders_count", "sum"),
            avg_promo_trip_share=("promo_trip_share", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_rides_last_90d=("rides_last_90d", "mean"),
            avg_response_rate=("responded_7d_rate", "mean"),
            total_ltv_180d=("ltv_180d", "sum"),
            total_margin=("total_contribution_margin", "sum"),
        )
        .reset_index()
    )
    summary["users_share"] = summary["users_count"] / total_users
    summary["avg_cancellation_rate"] = summary["cancelled_orders_total"] / summary["created_orders_total"].replace({0: np.nan})

    promo_dist = (
        segment_user_base.groupby(["compound_segment", "promo_dependency_segment"]).size().rename("cnt").reset_index()
    )
    if not promo_dist.empty:
        promo_dominant = promo_dist.sort_values(["compound_segment", "cnt"], ascending=[True, False]).drop_duplicates("compound_segment")
        summary = summary.merge(
            promo_dominant[["compound_segment", "promo_dependency_segment"]].rename(columns={"promo_dependency_segment": "dominant_promo_segment"}),
            on="compound_segment",
            how="left",
        )
    else:
        summary["dominant_promo_segment"] = pd.NA

    action_map = (
        segment_user_base.groupby("compound_segment")["recommended_action"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Observe / No immediate action")
        .reset_index()
    )
    summary = summary.merge(action_map, on="compound_segment", how="left")

    return summary.sort_values(["users_count", "avg_ltv_180d"], ascending=[False, False])


def get_segment_map_table(segment_user_base: pd.DataFrame) -> pd.DataFrame:
    if segment_user_base.empty:
        return pd.DataFrame(columns=["risk_segment", "value_segment", "users_count"])
    total_users = len(segment_user_base)
    grouped = (
        segment_user_base.groupby(["risk_segment", "value_segment"], dropna=False)
        .agg(
            users_count=("user_id", "nunique"),
            avg_ltv_180d=("ltv_180d", "mean"),
            created_orders_total=("created_orders_count", "sum"),
            cancelled_orders_total=("cancelled_orders_count", "sum"),
            avg_promo_trip_share=("promo_trip_share", "mean"),
            avg_response_rate=("responded_7d_rate", "mean"),
        )
        .reset_index()
    )
    grouped["users_share"] = grouped["users_count"] / total_users
    grouped["avg_cancellation_rate"] = grouped["cancelled_orders_total"] / grouped["created_orders_total"].replace({0: np.nan})

    action_map = (
        segment_user_base.groupby(["risk_segment", "value_segment"])["recommended_action"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Observe / No immediate action")
        .reset_index()
    )
    grouped = grouped.merge(action_map, on=["risk_segment", "value_segment"], how="left")
    return grouped


def get_selected_segment_profile(segment_user_base: pd.DataFrame, segment_name: str) -> dict:
    selected = segment_user_base.loc[segment_user_base["compound_segment"] == segment_name].copy()
    if selected.empty:
        return {}
    created_sum = selected["created_orders_count"].sum()
    completed_sum = selected["completed_orders_count"].sum()
    profile = {
        "compound_segment": segment_name,
        "risk_segment": selected["risk_segment"].iloc[0],
        "value_segment": selected["value_segment"].iloc[0],
        "users_count": int(selected["user_id"].nunique()),
        "users_share": float(selected["user_id"].nunique() / max(len(segment_user_base), 1)),
        "avg_ltv_180d": float(selected["ltv_180d"].mean()),
        "total_ltv_180d": float(selected["ltv_180d"].sum()),
        "avg_margin_per_completed_order": float(selected["avg_margin_per_completed_order"].mean()),
        "cancellation_rate": float(selected["cancelled_orders_count"].sum() / created_sum) if created_sum > 0 else np.nan,
        "promo_trip_share": float(selected["promo_trip_share"].mean()),
        "avg_recency_days": float(selected["recency_days"].mean()),
        "avg_rides_last_90d": float(selected["rides_last_90d"].mean()),
        "avg_response_rate": float(selected["responded_7d_rate"].mean()),
        "recommended_action": selected["recommended_action"].mode().iloc[0],
        "dominant_promo_segment": selected["promo_dependency_segment"].mode().iloc[0],
        "avg_created_orders": float(selected["created_orders_count"].mean()),
        "avg_completed_orders": float(selected["completed_orders_count"].mean()),
        "completed_orders_total": float(completed_sum),
        "created_orders_total": float(created_sum),
    }
    return profile


def compare_segment_to_baseline(segment_summary: pd.DataFrame, selected_segment: str, baseline_mode: str = "median") -> pd.DataFrame:
    if segment_summary.empty or selected_segment not in set(segment_summary["compound_segment"]) or segment_summary["compound_segment"].nunique() < 2:
        return pd.DataFrame(columns=["metric", "selected", "baseline", "delta"])

    metric_map = {
        "Users count": "users_count",
        "Users share": "users_share",
        "Avg LTV 180d": "avg_ltv_180d",
        "Total LTV 180d": "total_ltv_180d",
        "Avg margin per completed order": "avg_margin_per_completed_order",
        "Cancellation rate": "avg_cancellation_rate",
        "Promo trip share": "avg_promo_trip_share",
        "Avg recency days": "avg_recency_days",
        "Avg rides last 90d": "avg_rides_last_90d",
        "Avg response rate": "avg_response_rate",
    }

    selected_row = segment_summary.loc[segment_summary["compound_segment"] == selected_segment].iloc[0]
    baseline_row = segment_summary.median(numeric_only=True) if baseline_mode == "median" else segment_summary.mean(numeric_only=True)

    rows = []
    for label, col in metric_map.items():
        sel = float(selected_row[col]) if col in selected_row and pd.notna(selected_row[col]) else np.nan
        base = float(baseline_row[col]) if col in baseline_row and pd.notna(baseline_row[col]) else np.nan
        rows.append({"metric": label, "selected": sel, "baseline": base, "delta": sel - base})
    return pd.DataFrame(rows)


def get_selected_segment_charts_data(segment_user_base: pd.DataFrame, selected_segment: str) -> dict[str, pd.DataFrame]:
    selected = segment_user_base.loc[segment_user_base["compound_segment"] == selected_segment].copy()
    if selected.empty:
        empty = pd.DataFrame(columns=["metric", "selected", "baseline"])
        unavailable = pd.DataFrame(columns=["metric", "reason"])
        return {
            "key_metrics": empty,
            "monetary_metrics": empty,
            "ratio_metrics": empty,
            "unavailable_metrics": unavailable,
            "recency_rides": empty,
            "promo_margin": empty,
            "cancel_value": empty,
        }

    baseline = segment_user_base.loc[segment_user_base["compound_segment"] != selected_segment].copy()
    if baseline.empty:
        empty = pd.DataFrame(columns=["metric", "selected", "baseline"])
        unavailable = pd.DataFrame(columns=["metric", "reason"])
        return {
            "key_metrics": empty,
            "monetary_metrics": empty,
            "ratio_metrics": empty,
            "unavailable_metrics": unavailable,
            "recency_rides": empty,
            "promo_margin": empty,
            "cancel_value": empty,
        }

    selected_created_sum = float(selected["created_orders_count"].sum())
    baseline_created_sum = float(baseline["created_orders_count"].sum())
    selected_completed_sum = float(selected["completed_orders_count"].sum())
    baseline_completed_sum = float(baseline["completed_orders_count"].sum())

    metric_specs = [
        {
            "metric": "LTV 180d",
            "group": "monetary",
            "selected": float(selected["ltv_180d"].mean()),
            "baseline": float(baseline["ltv_180d"].mean()),
            "reason": None,
        },
        {
            "metric": "Avg margin per completed order",
            "group": "monetary",
            "selected": float(selected["avg_margin_per_completed_order"].mean()) if selected_completed_sum > 0 else np.nan,
            "baseline": float(baseline["avg_margin_per_completed_order"].mean()) if baseline_completed_sum > 0 else np.nan,
            "reason": "Нет созданных заказов" if (selected_completed_sum <= 0 or baseline_completed_sum <= 0) else None,
        },
        {
            "metric": "Cancellation rate",
            "group": "ratio",
            "selected": float(selected["cancelled_orders_count"].sum() / selected_created_sum) if selected_created_sum > 0 else np.nan,
            "baseline": float(baseline["cancelled_orders_count"].sum() / baseline_created_sum) if baseline_created_sum > 0 else np.nan,
            "reason": "Нет созданных заказов" if (selected_created_sum <= 0 or baseline_created_sum <= 0) else None,
        },
        {
            "metric": "Promo trip share",
            "group": "ratio",
            "selected": float(selected["promo_trip_share"].mean()) if selected_completed_sum > 0 else np.nan,
            "baseline": float(baseline["promo_trip_share"].mean()) if baseline_completed_sum > 0 else np.nan,
            "reason": "Нет созданных заказов" if (selected_completed_sum <= 0 or baseline_completed_sum <= 0) else None,
        },
        {
            "metric": "Response rate",
            "group": "ratio",
            "selected": float(selected["responded_7d_rate"].mean()),
            "baseline": float(baseline["responded_7d_rate"].mean()),
            "reason": "Нет маркетинговых касаний",
        },
    ]

    metric_rows: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, str]] = []
    for spec in metric_specs:
        row = {"metric": spec["metric"], "selected": spec["selected"], "baseline": spec["baseline"], "group": spec["group"]}
        if pd.notna(row["selected"]) and pd.notna(row["baseline"]):
            metric_rows.append(row)
        else:
            reason = spec["reason"] or "Недостаточно данных"
            unavailable_rows.append({"metric": spec["metric"], "reason": reason})

    key_metrics = pd.DataFrame(metric_rows, columns=["metric", "selected", "baseline", "group"])
    monetary_metrics = key_metrics.loc[key_metrics["group"] == "monetary", ["metric", "selected", "baseline"]].reset_index(drop=True)
    ratio_metrics = key_metrics.loc[key_metrics["group"] == "ratio", ["metric", "selected", "baseline"]].reset_index(drop=True)
    if not ratio_metrics.empty:
        ratio_metrics[["selected", "baseline"]] = ratio_metrics[["selected", "baseline"]] * 100.0
    unavailable_metrics = pd.DataFrame(unavailable_rows, columns=["metric", "reason"])

    recency_rides = pd.DataFrame(
        {
            "profile": ["Selected segment", "Baseline"],
            "recency_days": [selected["recency_days"].mean(), baseline["recency_days"].mean()],
            "rides_last_90d": [selected["rides_last_90d"].mean(), baseline["rides_last_90d"].mean()],
        }
    )

    promo_margin = pd.DataFrame(
        {
            "profile": ["Selected segment", "Baseline"],
            "promo_trip_share": [selected["promo_trip_share"].mean(), baseline["promo_trip_share"].mean()],
            "avg_margin_per_completed_order": [selected["avg_margin_per_completed_order"].mean(), baseline["avg_margin_per_completed_order"].mean()],
        }
    )

    cancel_value = pd.DataFrame(
        {
            "profile": ["Selected segment", "Baseline"],
            "cancellation_rate": [
                selected["cancelled_orders_count"].sum() / selected["created_orders_count"].sum() if selected["created_orders_count"].sum() > 0 else np.nan,
                baseline["cancelled_orders_count"].sum() / baseline["created_orders_count"].sum() if baseline["created_orders_count"].sum() > 0 else np.nan,
            ],
            "avg_ltv_180d": [selected["ltv_180d"].mean(), baseline["ltv_180d"].mean()],
        }
    )

    return {
        "key_metrics": key_metrics,
        "monetary_metrics": monetary_metrics,
        "ratio_metrics": ratio_metrics,
        "unavailable_metrics": unavailable_metrics,
        "recency_rides": recency_rides,
        "promo_margin": promo_margin,
        "cancel_value": cancel_value,
    }


def generate_segment_diagnostics(selected_profile: dict, baseline_profile: dict) -> list[str]:
    if not selected_profile or not baseline_profile:
        return ["Недостаточно данных для диагностической интерпретации выбранного сегмента."]

    def _fmt_delta(selected: float, baseline: float, percent: bool = False, suffix: str = "") -> str:
        if pd.isna(selected) or pd.isna(baseline):
            return "не рассчитывается"
        delta = selected - baseline
        sign = "+" if delta >= 0 else ""
        if percent:
            return f"{sign}{delta * 100:.1f} п.п."
        if suffix:
            return f"{sign}{delta:.1f} {suffix}"
        return f"{sign}{delta:.1f}"

    notes: list[str] = []
    ltv_delta = _fmt_delta(selected_profile.get("avg_ltv_180d", np.nan), baseline_profile.get("avg_ltv_180d", np.nan))
    promo_delta = _fmt_delta(selected_profile.get("promo_trip_share", np.nan), baseline_profile.get("promo_trip_share", np.nan), percent=True)
    cancel_delta = _fmt_delta(selected_profile.get("cancellation_rate", np.nan), baseline_profile.get("cancellation_rate", np.nan), percent=True)
    recency_delta = _fmt_delta(selected_profile.get("avg_recency_days", np.nan), baseline_profile.get("avg_recency_days", np.nan), suffix="дн.")
    rides_delta = _fmt_delta(selected_profile.get("avg_rides_last_90d", np.nan), baseline_profile.get("avg_rides_last_90d", np.nan))
    total_ltv_delta = _fmt_delta(selected_profile.get("total_ltv_180d", np.nan), baseline_profile.get("total_ltv_180d", np.nan))

    if selected_profile.get("avg_ltv_180d", np.nan) > baseline_profile.get("avg_ltv_180d", np.nan) and selected_profile.get("promo_trip_share", np.nan) > baseline_profile.get("promo_trip_share", np.nan):
        notes.append(f"LTV 180д выше эталона ({ltv_delta}), но доля промо-поездок также выше ({promo_delta}): экономика сегмента может быть частично промо-поддерживаемой.")
    if selected_profile.get("cancellation_rate", np.nan) > baseline_profile.get("cancellation_rate", np.nan):
        notes.append(f"Доля отмен выше эталона на {cancel_delta}; это операционный риск, который стоит проверять вместе с качеством supply/UX.")
    if selected_profile.get("avg_recency_days", np.nan) > baseline_profile.get("avg_recency_days", np.nan) and selected_profile.get("avg_rides_last_90d", np.nan) < baseline_profile.get("avg_rides_last_90d", np.nan):
        notes.append(f"Поведенческий профиль ослаблен: recency {recency_delta}, а поездок за 90 дней {rides_delta} относительно эталона.")
    if selected_profile.get("total_ltv_180d", np.nan) > baseline_profile.get("total_ltv_180d", np.nan) and selected_profile.get("avg_ltv_180d", np.nan) <= baseline_profile.get("avg_ltv_180d", np.nan):
        notes.append(f"Сегмент значим за счёт масштаба: суммарный LTV 180д выше на {total_ltv_delta}, при этом средний LTV на пользователя не выше эталона.")
    if selected_profile.get("avg_ltv_180d", np.nan) < baseline_profile.get("avg_ltv_180d", np.nan) and selected_profile.get("promo_trip_share", np.nan) > baseline_profile.get("promo_trip_share", np.nan):
        notes.append(f"Средний LTV 180д ниже эталона ({ltv_delta}) при повышенной промо-доле ({promo_delta}): сегмент экономически уязвим к стимулированию.")

    if not notes:
        notes.append(
            "Сегмент близок к эталону: "
            f"LTV 180д {ltv_delta}, доля отмен {cancel_delta}, доля промо-поездок {promo_delta}, recency {recency_delta}."
        )
    return notes[:5]


def get_segment_distribution_tables(segment_user_base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "risk_distribution": segment_user_base.groupby("risk_segment", dropna=False).agg(users_count=("user_id", "nunique")).reset_index(),
        "value_distribution": segment_user_base.groupby("value_segment", dropna=False).agg(users_count=("user_id", "nunique")).reset_index(),
        "promo_distribution": segment_user_base.groupby("promo_dependency_segment", dropna=False).agg(users_count=("user_id", "nunique")).reset_index(),
    }


def get_segment_priority_metrics(segment_summary: pd.DataFrame) -> pd.DataFrame:
    if segment_summary.empty:
        return segment_summary
    result = segment_summary.copy()
    result["segment_priority_flag"] = np.select(
        [
            (result["value_segment"] == "High value") & (result["risk_segment"].isin(["At risk", "Dormant"])),
            (result["value_segment"] == "Low value") & (result["dominant_promo_segment"] == "High promo dependency"),
        ],
        ["Retention priority", "Incentive control"],
        default="Monitor",
    )
    return result


def get_segment_kpis(segment_user_base: pd.DataFrame, segment_summary: pd.DataFrame) -> dict[str, float]:
    if segment_user_base.empty:
        return {
            "users_count": 0,
            "compound_segments_count": 0,
            "avg_ltv_180d": np.nan,
            "total_ltv_180d": 0.0,
            "avg_cancellation_rate": np.nan,
            "avg_promo_trip_share": np.nan,
            "avg_recency_days": np.nan,
            "risk_users_share": np.nan,
            "high_value_users_share": np.nan,
        }

    users_count = len(segment_user_base)
    risk_users = segment_user_base["risk_segment"].isin(["At risk", "Dormant"]).mean()
    high_value = segment_user_base["value_segment"].eq("High value").mean()
    created_sum = segment_user_base["created_orders_count"].sum()
    cancel_sum = segment_user_base["cancelled_orders_count"].sum()

    return {
        "users_count": users_count,
        "compound_segments_count": int(segment_summary["compound_segment"].nunique()) if not segment_summary.empty else 0,
        "avg_ltv_180d": float(segment_user_base["ltv_180d"].mean()),
        "total_ltv_180d": float(segment_user_base["ltv_180d"].sum()),
        "avg_cancellation_rate": float(cancel_sum / created_sum) if created_sum > 0 else np.nan,
        "avg_promo_trip_share": float(segment_user_base["promo_trip_share"].mean()),
        "avg_recency_days": float(segment_user_base["recency_days"].mean()),
        "risk_users_share": float(risk_users),
        "high_value_users_share": float(high_value),
    }


# Backward-compatible wrappers for legacy overview/segments code paths.
def build_segment_table(user_mart: pd.DataFrame) -> pd.DataFrame:
    base = build_segment_user_base(user_mart)
    summary = get_segment_summary(base)
    return summary.rename(columns={
        "users_count": "users",
        "avg_ltv_180d": "avg_ltv_180",
        "avg_rides_last_90d": "avg_trips_90d",
        "avg_response_rate": "avg_response_7d",
        "avg_cancellation_rate": "avg_cancel_rate",
        "dominant_promo_segment": "promo_dependency_segment",
    })


def build_risk_distribution(user_mart: pd.DataFrame) -> pd.DataFrame:
    base = build_segment_user_base(user_mart)
    return base.groupby("risk_segment", dropna=False).agg(users_count=("user_id", "nunique"), avg_ltv_180=("ltv_180d", "mean")).reset_index().rename(columns={"users_count": "users"})


def build_value_distribution(user_mart: pd.DataFrame) -> pd.DataFrame:
    base = build_segment_user_base(user_mart)
    return base.groupby("value_segment", dropna=False).agg(users_count=("user_id", "nunique"), avg_ltv_180=("ltv_180d", "mean")).reset_index().rename(columns={"users_count": "users"})


def build_risk_value_pivot(user_mart: pd.DataFrame, metric: str = "users") -> pd.DataFrame:
    table = get_segment_map_table(build_segment_user_base(user_mart)).copy()
    metric_map = {
        "users": "users_count",
        "avg_ltv_180": "avg_ltv_180d",
        "avg_cancel_rate": "avg_cancellation_rate",
        "promo_share": "avg_promo_trip_share",
        "avg_response_rate": "avg_response_rate",
    }
    value_col = metric_map.get(metric, metric)
    if value_col not in table.columns:
        value_col = "users_count"
    return table.pivot_table(index="risk_segment", columns="value_segment", values=value_col, fill_value=0)


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
