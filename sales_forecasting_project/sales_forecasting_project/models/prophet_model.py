"""
prophet_model.py
----------------
Fits a Facebook/Meta Prophet model on the daily sales series.

Prophet decomposes the series into trend + yearly seasonality + weekly
seasonality automatically, and we add `is_promotion` as an extra regressor
so the model can learn the sales lift caused by promotional campaigns.

Run:
    python models/prophet_model.py
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from prophet import Prophet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import compute_metrics, plot_forecast, print_metrics

DATA_PATH = "data/sales_data.csv"
TEST_DAYS = 60
OUT_PLOT = "outputs/prophet_forecast.png"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_prophet():
    df = load_data()

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y", "is_promotion"]]

    train = prophet_df.iloc[:-TEST_DAYS].copy()
    test = prophet_df.iloc[-TEST_DAYS:].copy()

    print(f"Training on {len(train)} days, testing on {len(test)} days")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",   # our seasonal effects scale with the trend
        changepoint_prior_scale=0.1,          # allow some trend flexibility
    )
    model.add_regressor("is_promotion")
    model.fit(train)

    future = test[["ds", "is_promotion"]].copy()
    forecast_df = model.predict(future)

    forecast = forecast_df["yhat"].values
    lower = forecast_df["yhat_lower"].values
    upper = forecast_df["yhat_upper"].values

    metrics = compute_metrics(test["y"].values, forecast)
    print_metrics("Prophet", metrics)

    os.makedirs("outputs", exist_ok=True)
    plot_forecast(
        train_dates=train["ds"].iloc[-180:],
        train_actual=train["y"].values[-180:],
        test_dates=test["ds"],
        test_actual=test["y"].values,
        forecast=forecast,
        model_name="Prophet",
        out_path=OUT_PLOT,
        forecast_lower=lower,
        forecast_upper=upper,
    )

    # Also save Prophet's own component breakdown (trend/weekly/yearly)
    fig = model.plot_components(forecast_df)
    fig.savefig("outputs/prophet_components.png", dpi=150)
    print("Saved plot -> outputs/prophet_components.png")

    return model, metrics


if __name__ == "__main__":
    run_prophet()
