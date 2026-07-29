"""
visualization.py
----------------
Shared plotting utilities for comparing forecast vs. actual sales,
and for computing standard forecast accuracy metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    """Return a dict of MAE, RMSE, MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_forecast(train_dates, train_actual, test_dates, test_actual, forecast,
                   model_name, out_path, forecast_lower=None, forecast_upper=None):
    """
    Plot historical training data, actual test-period sales, and the model's
    forecast for that same test period, optionally with a confidence band.
    """
    plt.figure(figsize=(13, 6))
    plt.plot(train_dates, train_actual, label="Training Actual Sales", color="#4C72B0", linewidth=1)
    plt.plot(test_dates, test_actual, label="Actual Sales (Test Period)", color="#2C3E50", linewidth=2)
    plt.plot(test_dates, forecast, label=f"{model_name} Forecast", color="#E74C3C",
              linewidth=2, linestyle="--")

    if forecast_lower is not None and forecast_upper is not None:
        plt.fill_between(test_dates, forecast_lower, forecast_upper, color="#E74C3C",
                          alpha=0.15, label="Confidence Interval")

    plt.axvline(x=test_dates.iloc[0] if hasattr(test_dates, "iloc") else test_dates[0],
                color="gray", linestyle=":", linewidth=1, label="Forecast Start")
    plt.title(f"{model_name}: Forecast vs. Actual Sales", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Sales ($)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


def print_metrics(model_name, metrics):
    print(f"\n{model_name} Accuracy Metrics")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:>6}: {v:,.3f}")
