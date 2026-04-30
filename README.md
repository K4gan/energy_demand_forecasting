# Energy demand forecasting

Hourly electricity demand forecasting project with synthetic weather, calendar effects, lag features and time-based validation. It is deliberately built as a compact forecasting workflow rather than a one-off notebook.

## What is inside

- Local hourly demand generator covering weather, weekend, holiday and business-hour effects.
- Feature engineering for lag values and rolling demand means.
- Three sklearn regressors: histogram gradient boosting, random forest and extra trees.
- Time-based holdout evaluation.
- CLI script for recursive next-hour forecasts.
- Notebook for demand seasonality and lag diagnostics.

## Dataset

Generated in `train.py` by `make_energy_series()`.

Columns:

| Column | Meaning |
| --- | --- |
| `timestamp` | Hourly timestamp |
| `temperature_c` | Synthetic observed temperature |
| `humidity` | Synthetic humidity |
| `wind_speed` | Synthetic wind speed |
| `hour`, `dayofweek`, `month` | Calendar features |
| `is_weekend`, `is_holiday` | Calendar flags |
| `demand_mw` | Target electricity demand |

Training adds demand lags (`1`, `2`, `3`, `24`, `48`, `168`) and rolling means (`24`, `168` hours).

## Model comparison

`python train.py` stores results in `models/forecast_metadata.json`.

| Model | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- |
| Hist gradient boosting | generated at train time | generated at train time | generated at train time | generated at train time |
| Random forest | generated at train time | generated at train time | generated at train time | generated at train time |
| Extra trees | generated at train time | generated at train time | generated at train time | generated at train time |

RMSE is used for model selection because large forecast errors are operationally expensive for short-term demand planning.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Forecast the next day:

```bash
python predict.py --model boosting --horizon 24
```

## Design notes

The validation split is time-based, not random. That keeps future observations out of training and gives a better estimate of how the model behaves when forecasting forward. The prediction script uses recursive forecasting: each predicted hour is appended to history so future lag features can be built.
