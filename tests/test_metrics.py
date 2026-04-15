
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
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "activated_flag": [True, False, True],
            "completed_trips": [3, 0, 2],
            "active_90d_flag": [True, False, False],
            "margin_180d": [100.0, 0.0, 50.0],
            "avg_trip_margin": [30.0, 0.0, 20.0],
            "acquisition_cost": [50.0, 40.0, 25.0],
        }
    )
    metrics = compute_overview_metrics(df)
    assert metrics["total_users"] == 3
    assert metrics["activated_users"] == 2
    assert metrics["completed_trips"] == 5
