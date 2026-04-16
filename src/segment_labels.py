from __future__ import annotations

import pandas as pd

RISK_ORDER = ["Stable / Active", "Cooling", "At risk", "Dormant"]
VALUE_ORDER = ["Low value", "Medium value", "High value"]
PROMO_ORDER = ["Low promo dependency", "Medium promo dependency", "High promo dependency"]

RISK_LABELS = {
    "Stable / Active": "Стабильные / активные",
    "Cooling": "Остывающие",
    "At risk": "В зоне риска",
    "Dormant": "Спящие",
}
VALUE_LABELS = {
    "Low value": "Низкая ценность",
    "Medium value": "Средняя ценность",
    "High value": "Высокая ценность",
}
PROMO_LABELS = {
    "Low promo dependency": "Низкая промо-зависимость",
    "Medium promo dependency": "Средняя промо-зависимость",
    "High promo dependency": "Высокая промо-зависимость",
}
ACTION_LABELS = {
    "Protect / Retain": "Удерживать",
    "Reactivate": "Реактивировать",
    "Stimulate carefully": "Стимулировать осторожно",
    "Limit incentives": "Ограничить субсидии",
    "Observe / No immediate action": "Наблюдать без немедленного действия",
}


def localize_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    if "risk_segment" in localized.columns:
        localized["risk_segment_ru"] = localized["risk_segment"].map(RISK_LABELS).fillna("недостаточно данных")
    if "value_segment" in localized.columns:
        localized["value_segment_ru"] = localized["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
    if "promo_dependency_segment" in localized.columns:
        localized["promo_dependency_segment_ru"] = localized["promo_dependency_segment"].map(PROMO_LABELS).fillna("недостаточно данных")
    if "recommended_action" in localized.columns:
        localized["recommended_action_ru"] = localized["recommended_action"].map(ACTION_LABELS).fillna("Наблюдать без немедленного действия")
    if "dominant_promo_segment" in localized.columns:
        localized["dominant_promo_segment_ru"] = localized["dominant_promo_segment"].map(PROMO_LABELS).fillna("недостаточно данных")
    if "compound_segment" in localized.columns and "risk_segment" in localized.columns and "value_segment" in localized.columns:
        localized["compound_segment_ru"] = (
            localized["risk_segment"].map(RISK_LABELS).fillna("недостаточно данных")
            + " × "
            + localized["value_segment"].map(VALUE_LABELS).fillna("недостаточно данных")
        )
    return localized
