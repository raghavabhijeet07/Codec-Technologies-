"""
main.py
-------
Runs the full pipeline end-to-end:
  1. Generates the synthetic sales dataset (if not already present)
  2. Fits SARIMA, Prophet, and LSTM models
  3. Saves individual forecast-vs-actual plots for each model
  4. Produces a combined comparison chart + metrics table

Run:
    python main.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def ensure_data():
    if not os.path.exists("data/sales_data.csv"):
        print("No dataset found — generating synthetic sales data...")
        from data.generate_data import generate_sales_data
        df = generate_sales_data()
        df.to_csv("data/sales_data.csv", index=False)
    else:
        print("Using existing data/sales_data.csv")


def main():
    ensure_data()
    os.makedirs("outputs", exist_ok=True)

    all_metrics = {}

    print("\n" + "=" * 60)
    print("1/3  Running SARIMA")
    print("=" * 60)
    from models.arima_model import run_arima
    arima_model, arima_metrics = run_arima()
    all_metrics["SARIMA"] = arima_metrics

    print("\n" + "=" * 60)
    print("2/3  Running Prophet")
    print("=" * 60)
    from models.prophet_model import run_prophet
    prophet_model, prophet_metrics = run_prophet()
    all_metrics["Prophet"] = prophet_metrics

    print("\n" + "=" * 60)
    print("3/3  Running LSTM")
    print("=" * 60)
    from models.lstm_model import run_lstm
    lstm_model, lstm_metrics = run_lstm()
    all_metrics["LSTM"] = lstm_metrics

    # Summary table
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics).T
    summary_df = summary_df.round(3)
    print(summary_df)
    summary_df.to_csv("outputs/model_comparison_metrics.csv")
    print("\nSaved metrics table -> outputs/model_comparison_metrics.csv")

    best_model = summary_df["MAPE"].idxmin()
    print(f"\nBest performing model (lowest MAPE): {best_model}")


if __name__ == "__main__":
    main()
