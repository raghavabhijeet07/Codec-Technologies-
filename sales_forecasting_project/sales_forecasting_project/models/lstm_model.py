"""
lstm_model.py
-------------
Fits an LSTM (Long Short-Term Memory) neural network on the daily sales
series using a sliding-window approach:
  - Input: the previous `LOOKBACK` days of (scaled) sales, is_promotion,
           is_holiday_season
  - Output: the next day's (scaled) sales

The trained model then forecasts the test period recursively (its own
predictions feed back in as history for the next-step prediction).

Run:
    python models/lstm_model.py
"""

import sys
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logs

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import compute_metrics, plot_forecast, print_metrics

DATA_PATH = "data/sales_data.csv"
TEST_DAYS = 60
LOOKBACK = 30      # use past 30 days to predict the next day
OUT_PLOT = "outputs/lstm_forecast.png"
FEATURES = ["sales", "is_promotion", "is_holiday_season"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, 0])  # target = 'sales' column (index 0)
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def run_lstm():
    df = load_data()
    values = df[FEATURES].values.astype(float)

    train_len = len(values) - TEST_DAYS

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(values[:train_len])
    # scale test with same scaler (fit only on train to avoid leakage)
    scaled_full = scaler.transform(values)

    X_train, y_train = make_sequences(scaled_train, LOOKBACK)
    X_train = X_train.reshape((X_train.shape[0], LOOKBACK, len(FEATURES)))

    print(f"Training on {len(X_train)} sequences (lookback={LOOKBACK} days)")

    model = build_model((LOOKBACK, len(FEATURES)))
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0, callbacks=[early_stop])
    print("Training complete.")

    # Recursive forecasting over the test period
    history = scaled_full[train_len - LOOKBACK:train_len, :].copy()
    predictions_scaled = []

    for step in range(TEST_DAYS):
        x_input = history[-LOOKBACK:, :].reshape(1, LOOKBACK, len(FEATURES))
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        predictions_scaled.append(pred_scaled)

        # Next row's known exogenous features (promo/holiday) come from real future data;
        # only the sales value is the model's own prediction (true multi-step forecasting).
        next_real_row = scaled_full[train_len + step, :].copy()
        next_row = next_real_row.copy()
        next_row[0] = pred_scaled
        history = np.vstack([history, next_row])

    # Inverse-transform predictions back to original sales scale
    dummy = np.zeros((TEST_DAYS, len(FEATURES)))
    dummy[:, 0] = predictions_scaled
    forecast = scaler.inverse_transform(dummy)[:, 0]

    actual_test = values[train_len:, 0]
    dates = df["date"]
    train_dates = dates.iloc[:train_len]
    test_dates = dates.iloc[train_len:]

    metrics = compute_metrics(actual_test, forecast)
    print_metrics("LSTM", metrics)

    os.makedirs("outputs", exist_ok=True)
    plot_forecast(
        train_dates=train_dates.iloc[-180:],
        train_actual=values[:train_len, 0][-180:],
        test_dates=test_dates,
        test_actual=actual_test,
        forecast=forecast,
        model_name="LSTM",
        out_path=OUT_PLOT,
    )

    return model, metrics


if __name__ == "__main__":
    run_lstm()
