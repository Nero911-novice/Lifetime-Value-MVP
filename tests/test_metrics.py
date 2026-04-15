
import pandas as pd

from src.metrics import (
    apply_common_filters,
    compute_overview_metrics,
    build_cohort_user_base,
    get_cohort_summary,
    get_cohort_maturity_table,
    build_retention_matrix,
    compare_cohort_to_baseline,
)


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


def test_get_cohort_maturity_table_handles_period_index_math():
    cohort_user_base = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "cohort_month": ["2025-01", "2025-02"],
        }
    )
    table = get_cohort_maturity_table(cohort_user_base, pd.Timestamp("2025-04-15"))
    assert set(table.columns) == {"cohort_month", "cohort_size", "maturity_months"}
    assert table["maturity_months"].min() >= 0
