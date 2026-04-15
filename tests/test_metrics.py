
import pandas as pd

from src.metrics import apply_common_filters, compute_overview_metrics


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
