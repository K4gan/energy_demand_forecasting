from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TARGET = "demand_mw"
LAGS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
FEATURES = [
    "temperature_c",
    "humidity",
    "wind_speed",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
]


def make_energy_series(days: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    timestamp = pd.date_range("2022-01-01", periods=days * 24, freq="h")
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    month = timestamp.month
    annual = np.sin(2 * np.pi * timestamp.dayofyear.to_numpy() / 365.25)
    daily = np.sin(2 * np.pi * (hour - 7) / 24)

    temperature = 16 + 11 * annual + 5 * daily + rng.normal(0, 2.1, len(timestamp))
    humidity = np.clip(58 - 0.9 * temperature + rng.normal(0, 8, len(timestamp)), 15, 95)
    wind_speed = np.clip(rng.gamma(2.2, 2.0, len(timestamp)), 0, 22)
    is_weekend = (dayofweek >= 5).astype(int)
    is_holiday = ((timestamp.month == 1) & (timestamp.day == 1) | (timestamp.month == 12) & (timestamp.day == 25)).astype(int)
    cooling = np.maximum(temperature - 22, 0)
    heating = np.maximum(12 - temperature, 0)
    business_hours = ((hour >= 8) & (hour <= 19) & (is_weekend == 0)).astype(int)

    demand = (
        980
        + 34 * cooling
        + 27 * heating
        + 85 * business_hours
        - 45 * is_weekend
        - 70 * is_holiday
        + 18 * np.sin(2 * np.pi * hour / 24)
        + 22 * np.sin(2 * np.pi * timestamp.dayofyear.to_numpy() / 365.25)
        + rng.normal(0, 28, len(timestamp))
    )
    demand = np.maximum(demand, 620)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperature_c": temperature.round(2),
            "humidity": humidity.round(2),
            "wind_speed": wind_speed.round(2),
            "hour": hour,
            "dayofweek": dayofweek,
            "month": month,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            TARGET: demand.round(2),
        }
    )


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.sort_values("timestamp").copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df[TARGET].shift(1).rolling(window).mean()
    return df.dropna().reset_index(drop=True)


def split_time(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    return train, test


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hourly energy demand forecasting models.")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=31)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw = make_energy_series(args.days, args.random_state)
    data = add_lag_features(raw)
    train, test = split_time(data, args.test_days)

    candidates = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=260, learning_rate=0.055, l2_regularization=0.05, random_state=args.random_state),
        "random_forest": RandomForestRegressor(n_estimators=260, min_samples_leaf=4, random_state=args.random_state, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=args.random_state, n_jobs=-1),
    }

    results = []
    for name, model in candidates.items():
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(test[FEATURES])
        row = {"model": name, **metrics(test[TARGET], prediction)}
        results.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    best = min(results, key=lambda row: row["rmse"])
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
        "best_model": best["model"],
        "metrics": results,
        "last_observed_rows": raw.tail(max(LAGS) + 200).to_dict(orient="records"),
    }
    with open(MODELS_DIR / "forecast_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print(pd.DataFrame(results).sort_values("rmse").to_string(index=False))
    print(f"\nBest model: {best['model']}")


if __name__ == "__main__":
    main()
