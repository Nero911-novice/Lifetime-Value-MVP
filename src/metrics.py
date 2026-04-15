
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


def compute_overview_metrics(user_mart: pd.DataFrame) -> dict:
    total_users = len(user_mart)
    activated_users = int(user_mart["activated_flag"].sum())
    completed_trips = int(user_mart["completed_trips"].sum())
    active_90d_share = float(user_mart["active_90d_flag"].mean()) if total_users else 0.0

    ltv_180_mean = float(user_mart.loc[user_mart["activated_flag"], "margin_180d"].mean()) if activated_users else 0.0
    trip_margin_mean = float(user_mart.loc[user_mart["completed_trips"] > 0, "avg_trip_margin"].mean()) if completed_trips else 0.0

    total_ltv_180 = float(user_mart["margin_180d"].sum())
    total_cac = float(user_mart["acquisition_cost"].sum())
    ltv_cac_ratio = total_ltv_180 / total_cac if total_cac > 0 else np.nan

    return {
        "total_users": total_users,
        "activated_users": activated_users,
        "activation_rate": activated_users / total_users if total_users else 0.0,
        "completed_trips": completed_trips,
        "active_90d_share": active_90d_share,
        "ltv_180_mean": ltv_180_mean,
        "avg_trip_margin": trip_margin_mean,
        "ltv_cac_ratio": ltv_cac_ratio,
    }


def build_overview_charts(user_mart: pd.DataFrame) -> dict:
    activated = user_mart.loc[user_mart["activated_flag"]].copy()
    cohort_trend = (
        activated.groupby("cohort_month", dropna=False)
        .agg(
            activated_users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
        )
        .reset_index()
        .sort_values("cohort_month")
    )

    channel_summary = (
        user_mart.groupby("acquisition_channel")
        .agg(
            users=("user_id", "count"),
            activation_rate=("activated_flag", "mean"),
            avg_ltv_180=("margin_180d", "mean"),
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
        )
        .reset_index()
        .sort_values("avg_ltv_180", ascending=False)
    )

    return {
        "cohort_trend": cohort_trend,
        "channel_summary": channel_summary,
        "city_summary": city_summary,
    }


def compute_cohort_matrices(user_mart: pd.DataFrame, trips: pd.DataFrame, max_age_months: int = 12) -> dict:
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

    maturity = (
        activated.assign(
            observation_month=user_mart["observation_date"].iloc[0].to_period("M").strftime("%Y-%m")
        )
        .groupby("cohort_month")["user_id"]
        .count()
        .rename("activated_users")
        .to_frame()
    )
    maturity["maturity_months"] = [
        (
            (pd.Period(str(user_mart["observation_date"].iloc[0].date()), freq="M").year - pd.Period(c, freq="M").year) * 12
            + (pd.Period(str(user_mart["observation_date"].iloc[0].date()), freq="M").month - pd.Period(c, freq="M").month)
        )
        for c in maturity.index
    ]
    maturity = maturity.reset_index()

    retention.columns = [f"M{int(col)}" for col in retention.columns]
    cumulative_margin.columns = [f"M{int(col)}" for col in cumulative_margin.columns]

    return {
        "retention_matrix": retention,
        "cohort_ltv_matrix": cumulative_margin,
        "cohort_maturity": maturity.sort_values("cohort_month"),
    }


def build_segment_table(user_mart: pd.DataFrame) -> pd.DataFrame:
    segment = (
        user_mart.groupby(["risk_segment", "value_segment", "promo_band"], dropna=False)
        .agg(
            users=("user_id", "count"),
            avg_ltv_180=("margin_180d", "mean"),
            avg_trips_90d=("recent_trips_90d", "mean"),
            avg_response_7d=("response_rate_7d", "mean"),
            active_90d_share=("active_90d_flag", "mean"),
        )
        .reset_index()
        .sort_values(["users", "avg_ltv_180"], ascending=[False, False])
    )
    return segment


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
        "Факт поездок, оборота и маржи",
        "История коммуникаций и response-подход",
        "Агрегация кампаний",
        "Документация полей",
    ]
    return result
