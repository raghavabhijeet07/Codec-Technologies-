"""
arima_model.py
--------------
Fits a SARIMA (Seasonal ARIMA) model on the daily sales series.

Why SARIMA and not plain ARIMA:
  Our sales data has weekly seasonality (weekends spike), so we use the
  seasonal component (order 7 = 7 days/week) on top of the standard
  (p, d, q) trend terms. pmdarima's auto_arima automatically searches for
  the best (p, d, q)(P, D, Q, m) combination via AIC minimization.

Run:
    python models/arima_model.py
"""

import sys
import os
import numpy as np
import pandas as pd
import pmdarima as pm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import compute_metrics, plot_forecast, print_metrics

DATA_PATH = "data/sales_data.csv"
TEST_DAYS = 60  # holdout period to evaluate forecast accuracy
OUT_PLOT = "outputs/arima_forecast.png"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_arima():
    df = load_data()
    series = df.set_index("date")["sales"]

    train = series.iloc[:-TEST_DAYS]
    test = series.iloc[-TEST_DAYS:]

    print(f"Training on {len(train)} days, testing on {len(test)} days")
    print("Running auto_arima to search for the best (p,d,q)(P,D,Q,7) parameters...")

    model = pm.auto_arima(
        train,
        seasonal=True,
        m=7,                     # weekly seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_P=2, max_Q=2,
        trace=False,
    )
    print("\nBest model order:", model.order, "seasonal order:", model.seasonal_order)

    forecast, conf_int = model.predict(n_periods=TEST_DAYS, return_conf_int=True)
    forecast = np.asarray(forecast)
    lower, upper = conf_int[:, 0], conf_int[:, 1]

    metrics = compute_metrics(test.values, forecast)
    print_metrics("SARIMA", metrics)

    os.makedirs("outputs", exist_ok=True)
    plot_forecast(
        train_dates=train.index[-180:],           # show last 180 training days for context
        train_actual=train.values[-180:],
        test_dates=test.index,
        test_actual=test.values,
        forecast=forecast,
        model_name="SARIMA",
        out_path=OUT_PLOT,
        forecast_lower=lower,
        forecast_upper=upper,
    )

    return model, metrics


if __name__ == "__main__":
    run_arima()
