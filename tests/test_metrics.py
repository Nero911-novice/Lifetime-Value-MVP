
import pandas as pd

from src.metrics import (
    apply_common_filters,
    compute_overview_metrics,
    build_overview_charts,
    build_cohort_user_base,
    get_cohort_summary,
    get_cohort_maturity_table,
    build_retention_matrix,
    compare_cohort_to_baseline,
    build_segment_user_base,
    get_segment_summary,
    get_segment_map_table,
    get_selected_segment_profile,
    compare_segment_to_baseline,
    get_selected_segment_charts_data,
    generate_segment_diagnostics,
)
from src.data_loader import load_demo_data, build_user_mart


def test_apply_common_filters_keeps_matching_city():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "home_city": ["Москва", "Казань"],
            "acquisition_channel": ["Органика", "Органика"],
            "preferred_tariff": ["Эконом", "Комфорт"],
            "activation_type": ["Органическая первая поездка", "Органическая первая поездка"],
            "activated_flag": [True, True],
        }
    )
    result = apply_common_filters(df, city="Москва", channel="Все", tariff="Все", activation_type="Все")
    assert len(result) == 1
    assert result.iloc[0]["user_id"] == "u1"


def test_compute_overview_metrics_returns_expected_counts():
    user_mart = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "activated_flag": [True, False, True],
            "completed_trips": [3, 0, 2],
            "completed_orders": [3, 0, 2],
            "cancelled_orders": [1, 0, 0],
            "total_orders": [4, 0, 2],
            "active_90d_flag": [True, False, False],
            "margin_180d": [100.0, 0.0, 50.0],
            "avg_trip_margin": [30.0, 0.0, 20.0],
            "acquisition_cost": [50.0, 40.0, 25.0],
            "first_trip_date": pd.to_datetime(["2025-01-10", None, "2025-01-15"]),
            "registration_date": pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-08"]),
            "observation_date": pd.to_datetime(["2025-02-01", "2025-02-01", "2025-02-01"]),
        }
    )
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "user_id": ["u1", "u1", "u1", "u1", "u3", "u3"],
            "request_ts": pd.to_datetime([
                "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-01-20", "2025-01-18"
            ]),
            "order_status": ["completed", "completed", "completed", "cancelled", "completed", "completed"],
            "contribution_margin": [10.0, 15.0, 5.0, 0.0, 11.0, 9.0],
            "gmv": [100.0, 120.0, 80.0, 0.0, 90.0, 95.0],
        }
    )
    metrics = compute_overview_metrics(user_mart, trips)
    assert metrics["total_users"] == 3
    assert metrics["activated_users"] == 2
    assert metrics["completed_orders"] == 5
    assert metrics["total_orders"] == 6
    assert round(metrics["cancel_rate"], 4) == round(1 / 6, 4)


def test_build_overview_charts_contains_recommended_action():
    user_mart = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "activated_flag": [True, True],
            "cohort_month": ["2025-01", "2025-01"],
            "margin_180d": [100.0, 220.0],
            "active_90d_flag": [True, False],
            "cancel_rate": [0.02, 0.25],
            "registration_date": pd.to_datetime(["2024-12-20", "2024-12-25"]),
            "risk_segment": ["Stable / Active", "At Risk"],
            "value_segment": ["High Value", "VIP"],
            "acquisition_channel": ["Органика", "Платная"],
            "acquisition_cost": [50.0, 80.0],
            "home_city": ["Москва", "Казань"],
            "promo_band": ["High", "Low"],
        }
    )
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2"],
            "user_id": ["u1", "u2"],
            "order_status": ["completed", "completed"],
            "request_ts": pd.to_datetime(["2025-01-10", "2025-01-12"]),
            "contribution_margin": [12.0, 20.0],
            "gmv": [100.0, 140.0],
        }
    )

    charts = build_overview_charts(user_mart, trips)
    assert "recommended_action" in charts["segment_action_map"].columns
    assert charts["segment_action_map"]["recommended_action"].notna().all()


def test_build_cohort_user_base_and_summary():
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "home_city": ["Москва", "Москва"],
            "acquisition_channel": ["Органика", "Платная"],
            "activation_type": ["Органическая первая поездка", "Промо-активация"],
            "preferred_tariff": ["Эконом", "Комфорт"],
            "acquisition_cost": [100, 150],
            "registration_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "first_trip_date": pd.to_datetime(["2025-01-05", "2025-02-07"]),
        }
    )
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3", "t4"],
            "user_id": ["u1", "u1", "u2", "u2"],
            "request_ts": pd.to_datetime(["2025-01-05", "2025-02-10", "2025-02-07", "2025-03-03"]),
            "order_status": ["completed", "completed", "completed", "cancelled"],
            "contribution_margin": [50.0, 40.0, 20.0, 0.0],
            "promo_discount": [0.0, 5.0, 0.0, 0.0],
            "platform_revenue": [100, 80, 70, 0],
            "refund_amount": [0.0, 0.0, 0.0, 0.0],
            "variable_ops_cost": [20.0, 15.0, 10.0, 0.0],
            "gmv": [300, 250, 200, 0],
        }
    )
    cohort_user_base = build_cohort_user_base(users, trips, pd.DataFrame())
    summary = get_cohort_summary(cohort_user_base)
    assert len(cohort_user_base) == 2
    assert set(summary["cohort_month"]) == {"2025-01", "2025-02"}
    assert summary["cohort_size"].sum() == 2


def test_retention_matrix_and_baseline_compare():
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "home_city": ["Москва", "Москва", "Москва"],
            "acquisition_channel": ["Органика", "Органика", "Органика"],
            "activation_type": ["Органическая первая поездка"] * 3,
            "preferred_tariff": ["Эконом", "Эконом", "Эконом"],
            "acquisition_cost": [100, 100, 100],
            "registration_date": pd.to_datetime(["2025-01-01"] * 3),
            "first_trip_date": pd.to_datetime(["2025-01-05", "2025-01-06", "2025-02-02"]),
        }
    )
    trips = pd.DataFrame(
        {
            "trip_id": ["t1", "t2", "t3", "t4"],
            "user_id": ["u1", "u2", "u1", "u3"],
            "request_ts": pd.to_datetime(["2025-01-05", "2025-01-06", "2025-02-01", "2025-02-02"]),
            "order_status": ["completed", "completed", "completed", "completed"],
            "contribution_margin": [10.0, 20.0, 30.0, 15.0],
            "promo_discount": [0.0, 0.0, 0.0, 0.0],
            "platform_revenue": [40, 50, 60, 45],
            "refund_amount": [0.0, 0.0, 0.0, 0.0],
            "variable_ops_cost": [5.0, 5.0, 5.0, 5.0],
            "gmv": [100, 100, 120, 90],
        }
    )
    cohort_user_base = build_cohort_user_base(users, trips, pd.DataFrame())
    summary = get_cohort_summary(cohort_user_base)
    matrix = build_retention_matrix(cohort_user_base, trips)
    compare = compare_cohort_to_baseline(summary, selected_cohort="2025-01")
    assert "M0" in matrix.columns
    assert "2025-01" in matrix.index
    assert not compare.empty


def test_segment_layer_outputs_expected_structures():
    user_mart = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "home_city": ["Москва"] * 4,
            "acquisition_channel": ["Органика", "Платная", "Платная", "Органика"],
            "activation_type": ["Органическая первая поездка", "Промо-активация", "Реактивация", "Органическая первая поездка"],
            "preferred_tariff": ["Эконом", "Комфорт", "Эконом", "Бизнес"],
            "registration_date": pd.to_datetime(["2025-01-01", "2025-01-03", "2024-12-20", "2025-02-01"]),
            "first_trip_date": pd.to_datetime(["2025-01-05", "2025-01-10", "2025-01-01", "2025-02-05"]),
            "total_orders": [12, 5, 1, 0],
            "completed_orders": [10, 3, 1, 0],
            "cancelled_orders": [2, 2, 0, 0],
            "margin_30d": [120.0, 30.0, 5.0, 0.0],
            "margin_90d": [360.0, 70.0, 8.0, 0.0],
            "margin_180d": [780.0, 180.0, 12.0, 0.0],
            "margin_365d": [1200.0, 260.0, 18.0, 0.0],
            "total_margin": [1500.0, 350.0, 20.0, 0.0],
            "recent_trips_30d": [6, 1, 0, 0],
            "recent_trips_90d": [9, 2, 0, 0],
            "recency_days": [6, 40, 150, None],
            "promo_trip_share": [0.05, 0.4, 0.9, 0.0],
            "refund_rate": [0.01, 0.03, 0.2, 0.0],
            "response_rate_7d": [0.05, 0.15, 0.45, 0.0],
            "active_90d_flag": [True, True, False, False],
            "acquisition_cost": [200, 180, 120, 90],
        }
    )
    touches = pd.DataFrame(
        {
            "touch_id": ["t1", "t2", "t3", "t4"],
            "user_id": ["u1", "u2", "u2", "u3"],
            "converted_within_7d_flag": [0, 1, 0, 1],
        }
    )
    trips = pd.DataFrame({"request_ts": pd.to_datetime(["2025-04-01"])})

    segment_user_base = build_segment_user_base(user_mart, trips, touches)
    summary = get_segment_summary(segment_user_base)
    map_table = get_segment_map_table(segment_user_base)

    assert not segment_user_base.empty
    assert {"value_segment", "risk_segment", "promo_dependency_segment", "compound_segment", "recommended_action"}.issubset(segment_user_base.columns)
    assert not summary.empty
    assert not map_table.empty

    selected = summary.iloc[0]["compound_segment"]
    profile = get_selected_segment_profile(segment_user_base, selected)
    compare = compare_segment_to_baseline(summary, selected)
    baseline = {
        "avg_ltv_180d": float(summary["avg_ltv_180d"].median()),
        "total_ltv_180d": float(summary["total_ltv_180d"].median()),
        "cancellation_rate": float(summary["avg_cancellation_rate"].median()),
        "promo_trip_share": float(summary["avg_promo_trip_share"].median()),
        "avg_recency_days": float(summary["avg_recency_days"].median()),
        "avg_rides_last_90d": float(summary["avg_rides_last_90d"].median()),
    }
    diagnostics = generate_segment_diagnostics(profile, baseline)

    assert profile["users_count"] >= 1
    assert not compare.empty
    assert len(diagnostics) >= 1


def test_segment_orders_fallback_uses_total_orders_when_count_fields_are_empty():
    user_mart = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "home_city": ["Москва", "Москва"],
            "city": ["Москва", "Москва"],
            "acquisition_channel": ["Органика", "Платная"],
            "activation_type": ["Органическая первая поездка", "Промо-активация"],
            "preferred_tariff": ["Эконом", "Эконом"],
            "total_orders": [10, 3],
            "completed_orders": [7, 2],
            "cancelled_orders": [3, 1],
            "created_orders_count": [0, 0],
            "completed_orders_count": [0, 0],
            "cancelled_orders_count": [0, 0],
            "margin_180d": [500.0, 200.0],
            "total_margin": [700.0, 260.0],
            "recent_trips_30d": [2, 1],
            "recent_trips_90d": [5, 2],
            "recency_days": [7, 44],
            "promo_trip_share": [0.15, 0.55],
            "response_rate_7d": [0.05, 0.25],
            "active_90d_flag": [True, True],
        }
    )
    base = build_segment_user_base(user_mart)
    assert base["created_orders_count"].tolist() == [10, 3]
    assert base["completed_orders_count"].tolist() == [7, 2]
    assert base["cancelled_orders_count"].tolist() == [3, 1]
    assert set(base["risk_segment"]) == {"Stable / Active", "Cooling"}


def test_demo_segment_distribution_is_not_collapsed():
    data = load_demo_data("data")
    user_mart = build_user_mart(data)
    base = build_segment_user_base(user_mart, data["trips"], data["marketing_touches"])
    assert base["value_segment"].nunique() >= 3
    assert base["risk_segment"].nunique() >= 3
    assert base["promo_dependency_segment"].nunique() >= 3
    assert base["compound_segment"].nunique() >= 6


def test_selected_segment_charts_split_monetary_and_ratio_and_handle_missing():
    segment_user_base = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "compound_segment": ["A", "A", "B", "B"],
            "ltv_180d": [100.0, 120.0, 80.0, 90.0],
            "avg_margin_per_completed_order": [20.0, 22.0, 18.0, 17.0],
            "created_orders_count": [5, 4, 6, 5],
            "completed_orders_count": [3, 2, 4, 3],
            "cancelled_orders_count": [2, 2, 2, 2],
            "promo_trip_share": [0.20, 0.10, 0.15, 0.05],
            "responded_7d_rate": [0.10, 0.20, 0.10, 0.30],
            "recency_days": [10, 12, 15, 16],
            "rides_last_90d": [4, 3, 2, 2],
        }
    )
    charts = get_selected_segment_charts_data(segment_user_base, selected_segment="A")
    assert charts["monetary_metrics"]["metric"].tolist() == ["LTV 180d", "Avg margin per completed order"]
    assert charts["ratio_metrics"]["metric"].tolist() == ["Cancellation rate", "Promo trip share", "Response rate"]
    assert charts["ratio_metrics"]["selected"].max() > 1.0  # values are converted to %
    assert charts["unavailable_metrics"].empty

    no_touches_base = segment_user_base.copy()
    no_touches_base["responded_7d_rate"] = pd.NA
    charts_missing = get_selected_segment_charts_data(no_touches_base, selected_segment="A")
    assert "Response rate" in charts_missing["unavailable_metrics"]["metric"].tolist()
    assert "Нет маркетинговых касаний" in charts_missing["unavailable_metrics"]["reason"].tolist()
