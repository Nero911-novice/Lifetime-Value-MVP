
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from .metrics import build_segment_user_base


DATA_FILES = {
    "users": "users.csv",
    "trips": "trips.csv",
    "marketing_touches": "marketing_touches.csv",
    "campaigns": "campaigns.csv",
    "data_dictionary": "data_dictionary.csv",
    "dataset_summary": "dataset_summary.csv",
}


def load_demo_data(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    path = Path(data_dir)
    users = pd.read_csv(
        path / DATA_FILES["users"],
        parse_dates=["registration_date", "first_app_open_date", "first_trip_date"],
    )
    trips = pd.read_csv(
        path / DATA_FILES["trips"],
        parse_dates=["request_ts", "completed_ts"],
    )
    touches = pd.read_csv(
        path / DATA_FILES["marketing_touches"],
        parse_dates=["touch_ts"],
    )
    campaigns = pd.read_csv(
        path / DATA_FILES["campaigns"],
        parse_dates=["start_ts", "end_ts"],
    )
    dictionary = pd.read_csv(path / DATA_FILES["data_dictionary"])
    dataset_summary = pd.read_csv(path / DATA_FILES["dataset_summary"])

    return {
        "users": users,
        "trips": trips,
        "marketing_touches": touches,
        "campaigns": campaigns,
        "data_dictionary": dictionary,
        "dataset_summary": dataset_summary,
    }


def build_user_mart(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    users = data["users"].copy()
    trips = data["trips"].copy()
    touches = data["marketing_touches"].copy()

    users["activated_flag"] = users["first_trip_date"].notna()
    observation_date = trips["request_ts"].max().normalize()

    order_agg = trips.groupby("user_id").agg(
        total_orders=("trip_id", "count"),
        completed_orders=("order_status", lambda s: int((s == "completed").sum())),
        cancelled_orders=("order_status", lambda s: int((s != "completed").sum())),
    ).reset_index()

    completed = trips.loc[trips["order_status"] == "completed"].copy()
    completed = completed.merge(
        users[["user_id", "first_trip_date"]],
        on="user_id",
        how="left",
    )
    completed["days_from_first_trip"] = (
        completed["request_ts"].dt.normalize() - completed["first_trip_date"]
    ).dt.days

    overall_trip_agg = completed.groupby("user_id").agg(
        completed_trips=("trip_id", "count"),
        total_gmv=("gmv", "sum"),
        total_margin=("contribution_margin", "sum"),
        total_platform_revenue=("platform_revenue", "sum"),
        avg_trip_margin=("contribution_margin", "mean"),
        avg_gmv=("gmv", "mean"),
        avg_rating=("rating", "mean"),
        last_trip_ts=("request_ts", "max"),
        refund_sum=("refund_amount", "sum"),
        refund_trip_count=("refund_amount", lambda s: int((s > 0).sum())),
        discounted_trip_count=("promo_discount", lambda s: int((s > 0).sum())),
        negative_margin_trip_count=("contribution_margin", lambda s: int((s < 0).sum())),
        avg_eta_minutes=("eta_minutes", "mean"),
    ).reset_index()

    time_windows = {}
    for horizon in (30, 90, 180, 365):
        frame = completed.loc[
            completed["days_from_first_trip"].between(0, horizon, inclusive="both")
        ].groupby("user_id").agg(
            **{
                f"trips_{horizon}d": ("trip_id", "count"),
                f"gmv_{horizon}d": ("gmv", "sum"),
                f"margin_{horizon}d": ("contribution_margin", "sum"),
            }
        ).reset_index()
        time_windows[horizon] = frame

    recent_windows = {}
    for recent in (30, 90):
        recent_cut = observation_date - pd.Timedelta(days=recent)
        frame = completed.loc[completed["request_ts"] >= recent_cut].groupby("user_id").agg(
            **{
                f"recent_trips_{recent}d": ("trip_id", "count"),
                f"recent_margin_{recent}d": ("contribution_margin", "sum"),
            }
        ).reset_index()
        recent_windows[recent] = frame

    touch_agg = touches.groupby("user_id").agg(
        total_touches=("touch_id", "count"),
        opened_touches=("opened_flag", "sum"),
        clicked_touches=("clicked_flag", "sum"),
        converted_touches_7d=("converted_within_7d_flag", "sum"),
        total_touch_cost=("touch_cost", "sum"),
        last_touch_ts=("touch_ts", "max"),
    ).reset_index()

    recent_touch_cut = observation_date - pd.Timedelta(days=90)
    touch_90 = touches.loc[touches["touch_ts"] >= recent_touch_cut].groupby("user_id").agg(
        touches_90d=("touch_id", "count"),
        converted_touches_90d=("converted_within_7d_flag", "sum"),
    ).reset_index()

    user_mart = users.copy()
    for frame in [order_agg, overall_trip_agg, touch_agg, touch_90, *time_windows.values(), *recent_windows.values()]:
        user_mart = user_mart.merge(frame, on="user_id", how="left")

    fill_zero_cols = [
        "total_orders",
        "completed_orders",
        "cancelled_orders",
        "completed_trips",
        "total_gmv",
        "total_margin",
        "total_platform_revenue",
        "avg_trip_margin",
        "avg_gmv",
        "refund_sum",
        "refund_trip_count",
        "discounted_trip_count",
        "negative_margin_trip_count",
        "total_touches",
        "opened_touches",
        "clicked_touches",
        "converted_touches_7d",
        "total_touch_cost",
        "touches_90d",
        "converted_touches_90d",
        "trips_30d",
        "gmv_30d",
        "margin_30d",
        "trips_90d",
        "gmv_90d",
        "margin_90d",
        "trips_180d",
        "gmv_180d",
        "margin_180d",
        "trips_365d",
        "gmv_365d",
        "margin_365d",
        "recent_trips_30d",
        "recent_margin_30d",
        "recent_trips_90d",
        "recent_margin_90d",
        "avg_eta_minutes",
    ]
    for col in fill_zero_cols:
        if col in user_mart:
            user_mart[col] = user_mart[col].fillna(0)

    user_mart["avg_rating"] = user_mart["avg_rating"].fillna(np.nan)
    user_mart["cohort_month"] = (
        user_mart["first_trip_date"].dt.to_period("M").astype("string").fillna("Не активирован")
    )
    user_mart["recency_days"] = (observation_date - user_mart["last_trip_ts"]).dt.days
    user_mart.loc[user_mart["completed_trips"] == 0, "recency_days"] = np.nan

    user_mart["active_90d_flag"] = user_mart["recent_trips_90d"] > 0
    user_mart["promo_trip_share"] = np.where(
        user_mart["completed_trips"] > 0,
        user_mart["discounted_trip_count"] / user_mart["completed_trips"],
        0.0,
    )
    user_mart["refund_rate"] = np.where(
        user_mart["completed_trips"] > 0,
        user_mart["refund_trip_count"] / user_mart["completed_trips"],
        0.0,
    )
    user_mart["cancel_rate"] = np.where(
        user_mart["total_orders"] > 0,
        user_mart["cancelled_orders"] / user_mart["total_orders"],
        0.0,
    )
    user_mart["completion_rate"] = np.where(
        user_mart["total_orders"] > 0,
        user_mart["completed_orders"] / user_mart["total_orders"],
        0.0,
    )
    user_mart["response_rate_7d"] = np.where(
        user_mart["total_touches"] > 0,
        user_mart["converted_touches_7d"] / user_mart["total_touches"],
        0.0,
    )
    user_mart["open_rate"] = np.where(
        user_mart["total_touches"] > 0,
        user_mart["opened_touches"] / user_mart["total_touches"],
        0.0,
    )
    user_mart["click_rate"] = np.where(
        user_mart["total_touches"] > 0,
        user_mart["clicked_touches"] / user_mart["total_touches"],
        0.0,
    )

    segment_base = build_segment_user_base(user_mart, trips, touches)
    user_mart = user_mart.merge(
        segment_base[["user_id", "risk_segment", "value_segment", "promo_dependency_segment"]],
        on="user_id",
        how="left",
    )

    user_mart["ltv_cac_180d"] = np.where(
        user_mart["acquisition_cost"] > 0,
        user_mart["margin_180d"] / user_mart["acquisition_cost"],
        np.nan,
    )

    user_mart["observation_date"] = observation_date

    return user_mart


def get_data_bundle(data_dir: str | Path) -> Dict[str, Any]:
    data = load_demo_data(data_dir)
    user_mart = build_user_mart(data)
    data["user_mart"] = user_mart
    return data
