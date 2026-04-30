from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import FEATURES, TARGET, add_lag_features, make_energy_series

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_FILES = {
    "boosting": "hist_gradient_boosting.joblib",
    "forest": "random_forest.joblib",
    "extra_trees": "extra_trees.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast next hourly energy demand values.")
    parser.add_argument("--model", choices=MODEL_FILES.keys(), default="boosting")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--random-state", type=int, default=31)
    return parser.parse_args()


def metadata_model_name(model_key: str) -> str:
    return MODEL_FILES[model_key].removesuffix(".joblib")


def weather_row_for_timestamp(timestamp: pd.Timestamp, random_state: int) -> dict[str, float | int | pd.Timestamp]:
    rng = np.random.default_rng(abs(hash((str(timestamp), random_state))) % (2**32))
    annual = np.sin(2 * np.pi * timestamp.dayofyear / 365.25)
    daily = np.sin(2 * np.pi * (timestamp.hour - 7) / 24)
    temperature = 16 + 11 * annual + 5 * daily + rng.normal(0, 1.6)
    humidity = float(np.clip(58 - 0.9 * temperature + rng.normal(0, 6), 15, 95))
    return {
        "timestamp": timestamp,
        "temperature_c": round(float(temperature), 2),
        "humidity": round(humidity, 2),
        "wind_speed": round(float(np.clip(rng.gamma(2.2, 2.0), 0, 22)), 2),
        "hour": int(timestamp.hour),
        "dayofweek": int(timestamp.dayofweek),
        "month": int(timestamp.month),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "is_holiday": int((timestamp.month == 1 and timestamp.day == 1) or (timestamp.month == 12 and timestamp.day == 25)),
        TARGET: 0.0,
    }


def main() -> None:
    args = parse_args()
    model_path = MODELS_DIR / MODEL_FILES[args.model]
    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}. Run python train.py first.")

    model = joblib.load(model_path)
    history = make_energy_series(args.days, args.random_state)
    rows = []

    for _ in range(args.horizon):
        next_time = history["timestamp"].max() + pd.Timedelta(hours=1)
        generated = pd.DataFrame([weather_row_for_timestamp(next_time, args.random_state)])
        generated[TARGET] = history[TARGET].iloc[-1]
        candidate = pd.concat([history, generated], ignore_index=True)
        features = add_lag_features(candidate).tail(1)
        forecast = float(model.predict(features[FEATURES])[0])
        generated[TARGET] = forecast
        history = pd.concat([history, generated], ignore_index=True)
        rows.append({"timestamp": str(next_time), "forecast_mw": round(forecast, 2)})

    metadata_path = MODELS_DIR / "forecast_metadata.json"
    label = metadata_model_name(args.model)
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        label = metadata.get("best_model", label) if label == metadata.get("best_model") else label

    print(f"model={label}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
