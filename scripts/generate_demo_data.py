
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)

CITIES = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск"]
CHANNELS = [
    "Органика",
    "Поиск платный",
    "Социальная реклама",
    "Реферальная программа",
    "Партнерский канал",
    "Брендовый трафик",
    "CRM реактивация",
]
TARIFFS = ["Эконом", "Комфорт", "Бизнес"]
PLATFORMS = ["iOS", "Android", "Web"]
SUBSCRIPTIONS = ["Нет", "Ride Pass"]
PAYMENT_METHODS = ["Карта", "Наличные", "Кошелек"]
WEATHER = ["Ясно", "Дождь", "Снег", "Пиковый спрос", "Праздник"]
TOUCH_CHANNELS = ["Push", "Email", "SMS", "In-App", "Retargeting"]
CAMPAIGN_TYPES = ["Welcome", "Discount", "Retention", "Reactivation", "Subscription", "Service"]
OFFER_TYPES = ["Без оффера", "5% скидка", "10% скидка", "15% скидка", "20% скидка", "Пробный Ride Pass"]
ORDER_STATUS = ["completed", "cancelled_rider", "cancelled_driver"]

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2026-03-31")


def random_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    delta = (end - start).days
    return start + pd.to_timedelta(int(RNG.integers(0, delta + 1)), unit="D")


def weighted_choice(values, probs):
    return values[int(RNG.choice(len(values), p=np.array(probs) / np.sum(probs)))]


def generate_users(n_users: int = 1800) -> pd.DataFrame:
    records = []
    for idx in range(1, n_users + 1):
        user_id = f"U{idx:05d}"
        reg_date = random_date(START_DATE, END_DATE - pd.Timedelta(days=5))
        channel = weighted_choice(CHANNELS, [0.22, 0.18, 0.16, 0.12, 0.1, 0.12, 0.1])
        city = weighted_choice(CITIES, [0.42, 0.2, 0.14, 0.13, 0.11])
        platform = weighted_choice(PLATFORMS, [0.43, 0.5, 0.07])
        preferred_tariff = weighted_choice(TARIFFS, [0.67, 0.25, 0.08])
        subscription_plan = weighted_choice(SUBSCRIPTIONS, [0.78, 0.22])
        payment_method = weighted_choice(PAYMENT_METHODS, [0.71, 0.14, 0.15])

        acquisition_cost = {
            "Органика": RNG.uniform(0, 35),
            "Брендовый трафик": RNG.uniform(20, 80),
            "Поиск платный": RNG.uniform(180, 520),
            "Социальная реклама": RNG.uniform(160, 470),
            "Реферальная программа": RNG.uniform(120, 320),
            "Партнерский канал": RNG.uniform(90, 260),
            "CRM реактивация": RNG.uniform(40, 140),
        }[channel]

        activate_prob = {
            "Органика": 0.83,
            "Брендовый трафик": 0.86,
            "Поиск платный": 0.8,
            "Социальная реклама": 0.76,
            "Реферальная программа": 0.88,
            "Партнерский канал": 0.73,
            "CRM реактивация": 0.58,
        }[channel]

        first_app_open_date = reg_date + pd.to_timedelta(int(RNG.integers(0, 4)), unit="D")
        activated = RNG.random() < activate_prob

        if not activated:
            first_trip_date = pd.NaT
            activation_type = "Не активирован"
            lifecycle_status = "Never Activated"
        else:
            delay = int(np.clip(RNG.gamma(2.0, 7.0), 0, 180))
            if channel == "CRM реактивация":
                delay += int(RNG.integers(7, 25))
            first_trip_date = reg_date + pd.to_timedelta(delay, unit="D")
            if first_trip_date > END_DATE:
                first_trip_date = pd.NaT
                activation_type = "Не активирован"
                lifecycle_status = "Never Activated"
                activated = False
            else:
                if channel == "Реферальная программа":
                    activation_type = "Реферальная первая поездка"
                elif channel == "CRM реактивация":
                    activation_type = "Реактивация после регистрации"
                elif delay > 30:
                    activation_type = "Отложенная активация"
                elif RNG.random() < 0.37:
                    activation_type = "Промо-первая поездка"
                else:
                    activation_type = "Органическая первая поездка"
                lifecycle_status = weighted_choice(
                    ["New", "Active", "At Risk", "Dormant", "Reactivated"],
                    [0.12, 0.46, 0.18, 0.14, 0.10],
                )

        promo_dependency_score = float(np.clip(RNG.normal(0.42, 0.23), 0.0, 1.0))
        if activation_type == "Промо-первая поездка":
            promo_dependency_score = float(np.clip(promo_dependency_score + 0.18, 0.0, 1.0))
        if channel == "CRM реактивация":
            promo_dependency_score = float(np.clip(promo_dependency_score + 0.12, 0.0, 1.0))

        records.append(
            {
                "user_id": user_id,
                "registration_date": reg_date.date().isoformat(),
                "first_app_open_date": first_app_open_date.date().isoformat(),
                "home_city": city,
                "registration_platform": platform,
                "acquisition_channel": channel,
                "acquisition_campaign": f"{channel[:3].upper()}_{int(RNG.integers(1, 18)):02d}",
                "acquisition_cost": round(float(acquisition_cost), 2),
                "activation_type": activation_type,
                "first_trip_date": first_trip_date.date().isoformat() if pd.notna(first_trip_date) else "",
                "preferred_tariff": preferred_tariff,
                "subscription_plan": subscription_plan,
                "payment_method_preference": payment_method,
                "promo_dependency_score": round(promo_dependency_score, 3),
                "multi_city_flag": bool(RNG.random() < 0.11),
                "support_contact_count_demo": int(RNG.integers(0, 5)),
                "fraud_risk_flag_demo": bool(RNG.random() < 0.015),
                "lifecycle_status_demo": lifecycle_status,
            }
        )

    users = pd.DataFrame.from_records(records)
    return users


def _base_trip_count(channel: str, activation_type: str, tariff: str, subscription_plan: str) -> int:
    base = {
        "Органика": 8,
        "Брендовый трафик": 9,
        "Поиск платный": 7,
        "Социальная реклама": 6,
        "Реферальная программа": 10,
        "Партнерский канал": 5,
        "CRM реактивация": 4,
    }[channel]
    if activation_type == "Промо-первая поездка":
        base -= 1
    if activation_type == "Реферальная первая поездка":
        base += 1
    if tariff == "Бизнес":
        base += 2
    if subscription_plan == "Ride Pass":
        base += 3
    return max(base, 1)


def generate_trips(users: pd.DataFrame) -> pd.DataFrame:
    records = []
    trip_counter = 1
    for _, user in users.iterrows():
        if not user["first_trip_date"]:
            # occasional cancelled requests even without completed rides
            if RNG.random() < 0.08:
                cancelled_ts = pd.Timestamp(user["registration_date"]) + pd.to_timedelta(int(RNG.integers(2, 40)), unit="D")
                records.append(
                    {
                        "trip_id": f"T{trip_counter:07d}",
                        "user_id": user["user_id"],
                        "request_ts": cancelled_ts.isoformat(),
                        "completed_ts": "",
                        "pickup_city": user["home_city"],
                        "dropoff_city": user["home_city"],
                        "tariff": user["preferred_tariff"],
                        "order_status": "cancelled_rider",
                        "distance_km": 0.0,
                        "duration_min": 0.0,
                        "surge_multiplier": 1.0,
                        "eta_minutes": round(float(RNG.uniform(4, 14)), 1),
                        "gmv": 0.0,
                        "platform_revenue": 0.0,
                        "promo_discount": 0.0,
                        "driver_bonus": 0.0,
                        "processing_cost": 0.0,
                        "insurance_cost": 0.0,
                        "support_cost": 0.0,
                        "refund_amount": 0.0,
                        "variable_ops_cost": 0.0,
                        "contribution_margin": 0.0,
                        "rating": np.nan,
                        "is_airport_trip": False,
                        "weather_bucket": weighted_choice(WEATHER, [0.48, 0.2, 0.08, 0.16, 0.08]),
                        "payment_method": user["payment_method_preference"],
                    }
                )
                trip_counter += 1
            continue

        first_trip_date = pd.Timestamp(user["first_trip_date"])
        observation_end = END_DATE
        active_days = max((observation_end - first_trip_date).days, 1)
        base_count = _base_trip_count(
            user["acquisition_channel"],
            user["activation_type"],
            user["preferred_tariff"],
            user["subscription_plan"],
        )
        trip_count = int(np.clip(RNG.negative_binomial(base_count, 0.35), 1, 60))
        if user["lifecycle_status_demo"] == "Dormant":
            trip_count = max(1, int(trip_count * 0.45))
        elif user["lifecycle_status_demo"] == "Active":
            trip_count = int(trip_count * 1.2)
        elif user["lifecycle_status_demo"] == "Reactivated":
            trip_count = max(2, int(trip_count * 0.85))
        if active_days < 45:
            trip_count = max(1, min(trip_count, int(active_days / 6) + 1))

        trip_dates = sorted(first_trip_date + pd.to_timedelta(RNG.integers(0, active_days + 1, size=trip_count), unit="D"))
        if user["lifecycle_status_demo"] == "Dormant" and len(trip_dates) > 3:
            # make a long pause after early activity
            trip_dates = trip_dates[: max(1, len(trip_dates) // 3)]
        if user["lifecycle_status_demo"] == "Reactivated" and len(trip_dates) > 4:
            gap_start = first_trip_date + pd.Timedelta(days=int(active_days * 0.35))
            gap_end = first_trip_date + pd.Timedelta(days=int(active_days * 0.72))
            filtered = [d for d in trip_dates if d < gap_start or d > gap_end]
            if len(filtered) >= 2:
                trip_dates = filtered

        for trip_dt in trip_dates:
            status = weighted_choice(ORDER_STATUS, [0.89, 0.07, 0.04])
            tariff = user["preferred_tariff"] if RNG.random() < 0.78 else weighted_choice(TARIFFS, [0.7, 0.22, 0.08])
            city = user["home_city"] if not user["multi_city_flag"] or RNG.random() < 0.82 else weighted_choice(CITIES, [0.4,0.2,0.14,0.13,0.13])
            distance = {
                "Эконом": RNG.gamma(3.2, 2.1),
                "Комфорт": RNG.gamma(3.6, 2.5),
                "Бизнес": RNG.gamma(4.2, 3.0),
            }[tariff]
            duration = distance * RNG.uniform(2.8, 4.4)
            weather = weighted_choice(WEATHER, [0.47, 0.2, 0.08, 0.16, 0.09])
            surge = 1.0
            if weather in ("Дождь", "Пиковый спрос"):
                surge += RNG.uniform(0.05, 0.45)
            if weather == "Праздник":
                surge += RNG.uniform(0.15, 0.55)

            request_ts = pd.Timestamp(trip_dt) + pd.to_timedelta(int(RNG.integers(6, 23)), unit="h") + pd.to_timedelta(int(RNG.integers(0, 60)), unit="m")
            eta = float(np.clip(RNG.normal(6.5, 2.5), 2.0, 18.0))
            airport = bool(RNG.random() < 0.11)
            if airport:
                distance *= RNG.uniform(1.8, 2.8)
                duration *= RNG.uniform(1.4, 2.2)

            base_fare = {"Эконом": 85, "Комфорт": 130, "Бизнес": 240}[tariff]
            per_km = {"Эконом": 24, "Комфорт": 31, "Бизнес": 46}[tariff]
            gmv = (base_fare + distance * per_km) * surge

            promo_discount = 0.0
            if user["activation_type"] == "Промо-первая поездка" and (request_ts - first_trip_date).days <= 14:
                promo_discount = gmv * RNG.uniform(0.12, 0.28)
            elif RNG.random() < max(0.05, min(0.45, user["promo_dependency_score"] * 0.38)):
                promo_discount = gmv * RNG.uniform(0.03, 0.18)

            if user["subscription_plan"] == "Ride Pass":
                promo_discount += min(gmv * 0.06, 90)

            commission_rate = {"Эконом": 0.22, "Комфорт": 0.24, "Бизнес": 0.28}[tariff]
            platform_revenue = gmv * commission_rate
            driver_bonus = 0.0
            if weather in ("Пиковый спрос", "Праздник") and RNG.random() < 0.42:
                driver_bonus = gmv * RNG.uniform(0.02, 0.07)
            processing_cost = max(6.0, gmv * 0.012)
            insurance_cost = max(4.0, gmv * 0.008)
            support_cost = 0.0
            if RNG.random() < 0.12:
                support_cost = RNG.uniform(8, 45)

            refund_amount = 0.0
            rating = np.nan
            completed_ts = ""
            if status == "completed":
                completed_ts = (request_ts + pd.to_timedelta(int(duration), unit="m")).isoformat()
                rating = round(float(np.clip(RNG.normal(4.76, 0.32), 2.0, 5.0)), 1)
                if rating < 4.2 and RNG.random() < 0.35:
                    refund_amount = gmv * RNG.uniform(0.1, 0.7)
                elif RNG.random() < 0.03:
                    refund_amount = gmv * RNG.uniform(0.05, 0.35)
            else:
                gmv = 0.0
                platform_revenue = 0.0
                promo_discount = 0.0
                driver_bonus = 0.0
                refund_amount = 0.0
                processing_cost = 0.0
                insurance_cost = 0.0
                support_cost = RNG.uniform(0, 6) if RNG.random() < 0.3 else 0.0
                distance = 0.0
                duration = 0.0

            variable_ops_cost = processing_cost + insurance_cost + support_cost
            contribution_margin = platform_revenue - promo_discount - driver_bonus - variable_ops_cost - refund_amount

            records.append(
                {
                    "trip_id": f"T{trip_counter:07d}",
                    "user_id": user["user_id"],
                    "request_ts": request_ts.isoformat(),
                    "completed_ts": completed_ts,
                    "pickup_city": city,
                    "dropoff_city": city if not airport else weighted_choice(CITIES, [0.42, 0.18, 0.14, 0.13, 0.13]),
                    "tariff": tariff,
                    "order_status": status,
                    "distance_km": round(float(distance), 2),
                    "duration_min": round(float(duration), 1),
                    "surge_multiplier": round(float(surge), 2),
                    "eta_minutes": round(float(eta), 1),
                    "gmv": round(float(gmv), 2),
                    "platform_revenue": round(float(platform_revenue), 2),
                    "promo_discount": round(float(promo_discount), 2),
                    "driver_bonus": round(float(driver_bonus), 2),
                    "processing_cost": round(float(processing_cost), 2),
                    "insurance_cost": round(float(insurance_cost), 2),
                    "support_cost": round(float(support_cost), 2),
                    "refund_amount": round(float(refund_amount), 2),
                    "variable_ops_cost": round(float(variable_ops_cost), 2),
                    "contribution_margin": round(float(contribution_margin), 2),
                    "rating": rating,
                    "is_airport_trip": airport,
                    "weather_bucket": weather,
                    "payment_method": user["payment_method_preference"],
                }
            )
            trip_counter += 1
    return pd.DataFrame.from_records(records)


def generate_touches(users: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    completed_trips = trips[trips["order_status"] == "completed"].copy()
    completed_trips["request_ts"] = pd.to_datetime(completed_trips["request_ts"])
    trip_lookup = completed_trips.groupby("user_id")["trip_id"].agg(list).to_dict()
    trip_time_lookup = completed_trips.groupby("user_id")["request_ts"].agg(list).to_dict()

    records = []
    touch_counter = 1
    for _, user in users.iterrows():
        user_id = user["user_id"]
        base_touches = int(np.clip(RNG.poisson(3 + user["promo_dependency_score"] * 4), 0, 14))
        if user["activation_type"] == "Не активирован":
            base_touches += int(RNG.integers(1, 4))
        if user["lifecycle_status_demo"] in ("At Risk", "Dormant", "Reactivated"):
            base_touches += int(RNG.integers(1, 5))

        for _ in range(base_touches):
            channel = weighted_choice(TOUCH_CHANNELS, [0.36, 0.2, 0.12, 0.18, 0.14])
            campaign_type = weighted_choice(CAMPAIGN_TYPES, [0.16, 0.26, 0.21, 0.18, 0.1, 0.09])
            offer = weighted_choice(OFFER_TYPES, [0.24, 0.2, 0.22, 0.18, 0.1, 0.06])
            campaign_id = f"{campaign_type[:3].upper()}_{int(RNG.integers(1, 40)):03d}"
            touch_ts = random_date(pd.Timestamp(user["registration_date"]), END_DATE)
            opened = bool(RNG.random() < {"Push": 0.42, "Email": 0.29, "SMS": 0.38, "In-App": 0.61, "Retargeting": 0.18}[channel])
            clicked = bool(opened and RNG.random() < 0.36)

            converted = False
            converted_trip_id = ""
            if user_id in trip_time_lookup:
                user_trip_times = trip_time_lookup[user_id]
                future_trip_indices = [i for i, ts in enumerate(user_trip_times) if ts >= touch_ts and ts <= touch_ts + pd.Timedelta(days=7)]
                base_conv = 0.08 + user["promo_dependency_score"] * 0.12
                if campaign_type in ("Reactivation", "Discount"):
                    base_conv += 0.05
                if clicked:
                    base_conv += 0.08
                if future_trip_indices and RNG.random() < min(base_conv, 0.65):
                    idx = int(future_trip_indices[0])
                    converted_trip_id = trip_lookup[user_id][idx]
                    converted = True

            treatment_group = weighted_choice(["control", "treatment_a", "treatment_b"], [0.18, 0.58, 0.24])
            records.append(
                {
                    "touch_id": f"M{touch_counter:07d}",
                    "user_id": user_id,
                    "touch_ts": touch_ts.isoformat(),
                    "touch_channel": channel,
                    "campaign_id": campaign_id,
                    "campaign_type": campaign_type,
                    "offer_type": offer,
                    "opened_flag": opened,
                    "clicked_flag": clicked,
                    "converted_within_7d_flag": converted,
                    "converted_trip_id": converted_trip_id,
                    "treatment_group_demo": treatment_group,
                    "touch_cost": round(float(RNG.uniform(0.8, 32.0)), 2),
                }
            )
            touch_counter += 1
    return pd.DataFrame.from_records(records)


def generate_campaigns(touches: pd.DataFrame) -> pd.DataFrame:
    grouped = touches.groupby(["campaign_id", "campaign_type", "touch_channel"], dropna=False).agg(
        touch_count=("touch_id", "count"),
        total_cost=("touch_cost", "sum"),
        conversions=("converted_within_7d_flag", "sum"),
        start_ts=("touch_ts", "min"),
        end_ts=("touch_ts", "max"),
    ).reset_index()

    grouped["campaign_name"] = grouped["campaign_type"] + " / " + grouped["touch_channel"] + " / " + grouped["campaign_id"]
    grouped["target_segment_demo"] = np.where(
        grouped["campaign_type"].isin(["Reactivation", "Retention"]),
        "Пользователи с риском оттока",
        np.where(grouped["campaign_type"] == "Welcome", "Новые пользователи", "Смешанный сегмент"),
    )
    grouped["conversion_rate_7d"] = (grouped["conversions"] / grouped["touch_count"]).round(4)
    grouped["budget_demo"] = grouped["total_cost"].round(2)
    grouped["start_ts"] = pd.to_datetime(grouped["start_ts"]).dt.date.astype(str)
    grouped["end_ts"] = pd.to_datetime(grouped["end_ts"]).dt.date.astype(str)
    return grouped[
        [
            "campaign_id",
            "campaign_name",
            "campaign_type",
            "touch_channel",
            "start_ts",
            "end_ts",
            "target_segment_demo",
            "touch_count",
            "conversions",
            "conversion_rate_7d",
            "budget_demo",
        ]
    ].sort_values("campaign_id")


def generate_data_dictionary() -> pd.DataFrame:
    rows = [
        ("users", "user_id", "string", "Уникальный идентификатор пользователя", "Связь таблиц и пользовательская карточка", "В демонстрации синтетический идентификатор."),
        ("users", "registration_date", "date", "Дата регистрации пользователя", "Когорта регистрации и путь до активации", "Не равна дате первой поездки."),
        ("users", "first_trip_date", "date", "Дата первой завершенной поездки", "Когорта активации и горизонт LTV", "Для неактивированных пользователей поле пустое."),
        ("users", "acquisition_channel", "category", "Источник первичного привлечения", "Срезы LTV и сравнение каналов", "В MVP укрупненная классификация."),
        ("users", "activation_type", "category", "Тип первой активации", "Интерпретация качества старта пользователя", "В боевой версии должен вычисляться по реальным событиям."),
        ("users", "acquisition_cost", "float", "Стоимость привлечения пользователя", "LTV/CAC и окупаемость", "В демонстрации задана синтетически."),
        ("users", "promo_dependency_score", "float", "Оценка зависимости от промо", "Сегментация и интерпретация маржинальности", "В будущем лучше считать модельно."),
        ("trips", "trip_id", "string", "Уникальный идентификатор заказа", "Детализация поездок и связь с маркетингом", "Один заказ = одна запись."),
        ("trips", "order_status", "category", "Статус заказа", "Фильтрация completed/cancelled", "В MVP без подробной событийной ленты."),
        ("trips", "gmv", "float", "Валовая стоимость поездки", "Оборот и валовая экономическая база", "Не равен выручке платформы."),
        ("trips", "platform_revenue", "float", "Доход платформы до скидок и переменных затрат", "Расчет маржи", "В демонстрации задан через упрощенную ставку комиссии."),
        ("trips", "promo_discount", "float", "Скидка пользователю", "Оценка промо-зависимости и маржи", "В боевой версии нужен источник скидки и тип субсидии."),
        ("trips", "driver_bonus", "float", "Субсидия или бонус водителю", "Contribution margin", "В MVP показан укрупненно."),
        ("trips", "variable_ops_cost", "float", "Переменные операционные затраты", "Contribution margin", "Включают процессинг, страхование и часть поддержки."),
        ("trips", "refund_amount", "float", "Возврат пользователю", "Корректировка фактической ценности поездки", "Может относиться к поездке с лагом."),
        ("trips", "contribution_margin", "float", "Маржинальный вклад поездки", "Базовая денежная сущность LTV", "В MVP без распределения фиксированных расходов."),
        ("marketing_touches", "touch_id", "string", "Уникальный идентификатор маркетингового касания", "Анализ коммуникаций", "В боевой системе нужны реальные идентификаторы событий."),
        ("marketing_touches", "touch_channel", "category", "Канал коммуникации", "Срезы response и реактивации", "Классификация укрупнена."),
        ("marketing_touches", "campaign_type", "category", "Тип кампании", "Интерпретация роли касания", "Нужна единая кампания-иерархия."),
        ("marketing_touches", "converted_within_7d_flag", "bool", "Флаг поездки в течение 7 дней после касания", "Демонстрация response-подхода", "Это не причинный эффект."),
        ("campaigns", "conversion_rate_7d", "float", "Доля касаний с поездкой в течение 7 дней", "Пилотная оценка отклика", "Не заменяет uplift и A/B-анализ."),
    ]
    return pd.DataFrame(rows, columns=["table_name", "field_name", "data_type", "description", "used_for", "limitations"])


def main(output_dir: str = "data") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    users = generate_users()
    trips = generate_trips(users)
    touches = generate_touches(users, trips)
    campaigns = generate_campaigns(touches)
    dictionary = generate_data_dictionary()

    users.to_csv(output_path / "users.csv", index=False)
    trips.to_csv(output_path / "trips.csv", index=False)
    touches.to_csv(output_path / "marketing_touches.csv", index=False)
    campaigns.to_csv(output_path / "campaigns.csv", index=False)
    dictionary.to_csv(output_path / "data_dictionary.csv", index=False)

    summary = pd.DataFrame(
        {
            "table_name": ["users", "trips", "marketing_touches", "campaigns", "data_dictionary"],
            "rows": [len(users), len(trips), len(touches), len(campaigns), len(dictionary)],
            "columns": [users.shape[1], trips.shape[1], touches.shape[1], campaigns.shape[1], dictionary.shape[1]],
        }
    )
    summary.to_csv(output_path / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
